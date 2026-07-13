"""
IEEE 33-Bus Engine — BFS Once Per Query
========================================
Architecture:
  1. run_bfs(changes)         — pandapower BFS, called exactly once per query
  2. compute_all_fast(bfs)    — all sections using only the BFS result (instant)
  3. compute_background(bfs)  — S3 + S4 perturbation loops (slow, run in bg)
"""

from __future__ import annotations
import time
import numpy as np
import networkx as nx

BRANCH_DATA = [
    (1,  2,  0.0922, 0.0477,   0,   0),
    (2,  3,  0.4930, 0.2511, 100,  60),
    (3,  4,  0.3660, 0.1864,  90,  40),
    (4,  5,  0.3811, 0.1941, 120,  80),
    (5,  6,  0.8190, 0.7070,  60,  30),
    (6,  7,  0.1872, 0.6188,  60,  20),
    (7,  8,  1.7114, 1.2351, 200, 100),
    (8,  9,  1.0300, 0.7400, 200, 100),
    (9, 10,  1.0400, 0.7400,  60,  20),
    (10,11,  0.1966, 0.0650,  60,  20),
    (11,12,  0.3744, 0.1238,  45,  30),
    (12,13,  1.4680, 1.1550,  60,  35),
    (13,14,  0.5416, 0.7129,  60,  35),
    (14,15,  0.5910, 0.5260, 120,  80),
    (15,16,  0.7463, 0.5450,  60,  10),
    (16,17,  1.2890, 1.7210,  60,  20),
    (17,18,  0.7320, 0.5740,  60,  20),
    (2, 19,  0.1640, 0.1565,  90,  40),
    (19,20,  1.5042, 1.3554,  90,  40),
    (20,21,  0.4095, 0.4784,  90,  40),
    (21,22,  0.7089, 0.9373,  90,  40),
    (3, 23,  0.4512, 0.3083,  90,  40),
    (23,24,  0.8980, 0.7091,  90,  50),
    (24,25,  0.8960, 0.7011, 420, 200),
    (6, 26,  0.2030, 0.1034, 420, 200),
    (26,27,  0.2842, 0.1447,  60,  25),
    (27,28,  1.0590, 0.9337,  60,  25),
    (28,29,  0.8042, 0.7006,  60,  20),
    (29,30,  0.5075, 0.2585, 120,  70),
    (30,31,  0.9744, 0.9630, 200, 600),
    (31,32,  0.3105, 0.3619, 150,  70),
    (32,33,  0.3410, 0.5302, 210, 100),
]

N = 33
BASE_KV = 12.66
I_RATING = 400

P0 = np.zeros(N)
Q0 = np.zeros(N)
for _fb, _tb, _R, _X, _P, _Q in BRANCH_DATA:
    P0[_tb-1] = _P
    Q0[_tb-1] = _Q

SCENARIOS = {
    "base":             {"label":"Base case",             "description":"Standard IEEE 33-bus loading — 3655 kW.",    "changes":{}},
    "load_spike_bus18": {"label":"Load spike Bus 18 +50%","description":"Bus 18 load increased 50%.",                 "changes":{"bus_load_scale":{18:1.5}}},
    "load_spike_bus25": {"label":"Load spike Bus 25 +80%","description":"Bus 25 load increased 80%.",                 "changes":{"bus_load_scale":{25:1.8}}},
    "dg_bus18":         {"label":"DG Bus 18 (500 kW)",    "description":"500 kW PV injected at bus 18.",              "changes":{"dg":{18:{"p_mw":0.5,"q_mvar":0.0}}}},
    "dg_bus33":         {"label":"DG Bus 33 (400 kW)",    "description":"400 kW DG at bus 33.",                       "changes":{"dg":{33:{"p_mw":0.4,"q_mvar":0.0}}}},
    "cap_bus18":        {"label":"Capacitor Bus 18 600kVAR","description":"Shunt capacitor at bus 18.",               "changes":{"capacitor":{18:0.6}}},
    "cap_bus30":        {"label":"Capacitor Bus 30 900kVAR","description":"Reactive compensation at bus 30.",         "changes":{"capacitor":{30:0.9}}},
    "outage_bus25":     {"label":"Bus 25 outage",          "description":"Bus 25 de-energised.",                      "changes":{"bus_load_scale":{25:0.0}}},
    "heavy_load":       {"label":"System load +30%",       "description":"All loads scaled up 30%.",                  "changes":{"global_scale":1.3}},
    "light_load":       {"label":"System load -40%",       "description":"All loads scaled down 40%.",                "changes":{"global_scale":0.6}},
}

BUS_POSITIONS = {
    1:(0,0),2:(1,0),3:(2,0),4:(3,0),5:(4,0),6:(5,0),
    7:(6,-1),8:(7,-1),9:(8,-1),10:(9,-1),11:(10,-1),12:(11,-1),
    13:(12,-1),14:(13,-1),15:(14,-1),16:(15,-1),17:(16,-1),18:(17,-1),
    19:(1,-2),20:(2,-2),21:(3,-2),22:(4,-2),
    23:(2,-3),24:(3,-3),25:(4,-3),
    26:(5,-4),27:(6,-4),28:(7,-4),29:(8,-4),
    30:(9,-4),31:(10,-4),32:(11,-4),33:(12,-4),
}


def _norm01(x):
    r=x-x.min(); d=r.max()+1e-12; return r/d

def _build_graph():
    G=nx.DiGraph(); cumZ=np.zeros(N)
    for fb,tb,R,X,*_ in BRANCH_DATA:
        Z=np.sqrt(R**2+X**2); G.add_edge(fb,tb,R=R,X=X,Z=Z); cumZ[tb-1]=cumZ[fb-1]+Z
    return G,cumZ

def _ser(obj):
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return float(obj)
    if isinstance(obj,dict): return {k:_ser(v) for k,v in obj.items()}
    if isinstance(obj,list): return [_ser(v) for v in obj]
    return obj


# ── BFS once ──────────────────────────────────────────────────────────────────
def run_bfs(changes: dict) -> dict:
    import pandapower as pp
    p=P0.copy(); q=Q0.copy()
    g=changes.get("global_scale",1.0); p*=g; q*=g
    for bus,scale in changes.get("bus_load_scale",{}).items():
        p[bus-1]=P0[bus-1]*scale; q[bus-1]=Q0[bus-1]*scale
    net=pp.create_empty_network(sn_mva=100)
    for i in range(N): pp.create_bus(net,vn_kv=BASE_KV,name=f"Bus {i+1}")
    pp.create_ext_grid(net,bus=0,vm_pu=1.0)
    for idx,(fb,tb,R,X,*_) in enumerate(BRANCH_DATA):
        pp.create_line_from_parameters(net,from_bus=fb-1,to_bus=tb-1,length_km=1.0,
            r_ohm_per_km=R,x_ohm_per_km=X,c_nf_per_km=0,max_i_ka=I_RATING/1000,name=f"L{idx}")
    for i in range(1,N): pp.create_load(net,bus=i,p_mw=p[i]/1000,q_mvar=q[i]/1000)
    for bus,dg in changes.get("dg",{}).items():
        pp.create_sgen(net,bus=bus-1,p_mw=dg["p_mw"],q_mvar=dg["q_mvar"])
    for bus,qv in changes.get("capacitor",{}).items():
        pp.create_shunt(net,bus=bus-1,q_mvar=-qv,p_mw=0.0)
    pp.runpp(net,algorithm="bfsw",max_iteration=200,tolerance_mva=1e-9)
    return dict(
        V_pu=net.res_bus.vm_pu.values.copy(),
        V_ang=net.res_bus.va_degree.values.copy(),
        Im=(net.res_line.i_ka.values*1000).copy(),
        loading=net.res_line.loading_percent.values.copy(),
        Pl=(net.res_line.pl_mw.values*1000).copy(),
        Ql=(net.res_line.ql_mvar.values*1000).copy(),
        totPl=float(net.res_line.pl_mw.values.sum()*1000),
        totQl=float(net.res_line.ql_mvar.values.sum()*1000),
        p_load=p, q_load=q,
    )


# ── Fast sections ─────────────────────────────────────────────────────────────
def compute_s1(base_bfs,scen_bfs,branch_labels):
    b,a=base_bfs,scen_bfs; V=a["V_pu"]
    VSI_b,VSI_a=[],[]
    for idx,(fb,tb,R,X,*_) in enumerate(BRANCH_DATA):
        for bfs,lst in [(b,VSI_b),(a,VSI_a)]:
            vf=bfs["V_pu"][fb-1]; pl=bfs["Pl"][idx]/1000; ql=bfs["Ql"][idx]/1000
            lst.append(float(4*(R*pl+X*ql)**2/max(vf**4,1e-9)))
    return _ser({
        "before":{"V_pu":b["V_pu"],"V_ang":b["V_ang"],"Im":b["Im"],
                  "loading":b["loading"],"Pl":b["Pl"],"Ql":b["Ql"],"VSI":VSI_b,
                  "totPl":b["totPl"],"totQl":b["totQl"],
                  "min_V":float(b["V_pu"].min()),"min_V_bus":int(b["V_pu"].argmin())+1,
                  "max_loading":float(b["loading"].max()),
                  "buses_below_095":int((b["V_pu"]<0.95).sum())},
        "after": {"V_pu":a["V_pu"],"V_ang":a["V_ang"],"Im":a["Im"],
                  "loading":a["loading"],"Pl":a["Pl"],"Ql":a["Ql"],"VSI":VSI_a,
                  "totPl":a["totPl"],"totQl":a["totQl"],
                  "min_V":float(V.min()),"min_V_bus":int(V.argmin())+1,
                  "max_loading":float(a["loading"].max()),
                  "buses_below_095":int((V<0.95).sum()),
                  "branch_labels":branch_labels},
        "delta": {"dV":(a["V_pu"]-b["V_pu"]).tolist(),
                  "dLoad":(a["loading"]-b["loading"]).tolist(),
                  "dPl":(a["Pl"]-b["Pl"]).tolist(),
                  "d_min_V":round(float(V.min()-b["V_pu"].min()),4),
                  "d_totPl":round(a["totPl"]-b["totPl"],2),
                  "d_totQl":round(a["totQl"]-b["totQl"],2),
                  "d_max_loading":round(float(a["loading"].max()-b["loading"].max()),1),
                  "d_buses_below":int((V<0.95).sum())-int((b["V_pu"]<0.95).sum())},
    })

def compute_s2(bfs):
    G,cumZ=_build_graph(); G_ug=G.to_undirected()
    for u,v,d in G_ug.edges(data=True): d["weight"]=d["Z"]
    depth=dict(nx.single_source_shortest_path_length(G_ug,1))
    depth_arr=np.array([depth.get(b,0) for b in range(1,N+1)],dtype=float)
    dist_mat=np.zeros((N,N))
    for i in range(1,N+1):
        lengths=nx.single_source_dijkstra_path_length(G_ug,i,weight="weight")
        for j in range(1,N+1): dist_mat[i-1,j-1]=lengths.get(j,0.0)
    V=bfs["V_pu"]; r_ZV=float(np.corrcoef(cumZ,V)[0,1])
    return _ser({"cumulative_Z_ohm":cumZ,"feeder_depth_hops":depth_arr,
                 "elec_dist_matrix":dist_mat,"V_pu":V,
                 "summary":{"r_Z_Vpu":round(r_ZV,4),
                             "max_elec_dist_bus":int(cumZ.argmax())+1,
                             "max_elec_dist_ohm":round(float(cumZ.max()),4)}})

def compute_s5(bfs):
    V=bfs["V_pu"]; _,cumZ=_build_graph()
    zones={"near_slack":[2,3,4,19],"mid_feeder":[7,8,11,26,27],"far_end":[17,18,22,25,33]}
    return _ser({
        "zone_buses":zones,
        "zone_voltages":{z:[float(V[b-1]) for b in buses] for z,buses in zones.items()},
        "zone_mean_V":{z:round(float(np.mean([V[b-1] for b in buses])),4) for z,buses in zones.items()},
        "feeder_end_buses":{b:{"V_pu":round(float(V[b-1]),4),"below_095":float(V[b-1])<0.95} for b in [18,25,33]},
        "attenuation_proxy":{f"bus_{b}":{"cumZ":float(cumZ[b-1]),"V_pu":round(float(V[b-1]),4)} for b in [2,8,18,24,30]},
        "base_V18":round(float(V[17]),4),
    })

def compute_s6(bfs,s3_data=None):
    V=bfs["V_pu"]; G,cumZ=_build_graph(); G_ug=G.to_undirected()
    for u,v,d in G_ug.edges(data=True): d["weight"]=1/(d["Z"]+1e-6)
    cc_arr=np.array([nx.closeness_centrality(G_ug)[b] for b in range(1,N+1)])
    bc_arr=np.array([nx.betweenness_centrality(G_ug,weight="weight")[b] for b in range(1,N+1)])
    inv_Z=_norm01(1.0/(cumZ+0.01))
    BII=np.array(s3_data["BII"]) if s3_data else _norm01(cumZ)*0.05
    VFI=np.array(s3_data["VFI"]) if s3_data else _norm01(1-V)*0.05
    ECI=0.30*_norm01(cc_arr)+0.30*inv_Z+0.25*_norm01(BII)+0.15*_norm01(VFI)
    dVdQ_proxy=_norm01((1-V)/(cumZ+0.1))
    instability=_norm01(_norm01(VFI)+dVdQ_proxy+_norm01(1-V))
    DPF=_norm01(cumZ*(1-V))
    top10=np.argsort(ECI)[::-1][:10]
    return _ser({"BII":BII,"VFI":VFI,"ECI":ECI,"instability":instability,"DPF":DPF,
                 "betweenness":bc_arr,"closeness":cc_arr,
                 "top10_by_ECI":{"bus_numbers":(top10+1).tolist(),"ECI":ECI[top10].tolist(),
                                  "BII":BII[top10].tolist(),"VFI":VFI[top10].tolist(),
                                  "instability":instability[top10].tolist(),"V_pu":V[top10].tolist()},
                 "summary":{"top5_ECI":(np.argsort(ECI)[::-1][:5]+1).tolist(),
                             "top5_instability":(np.argsort(instability)[::-1][:5]+1).tolist()},
                 "is_proxy":s3_data is None})

def compute_s9(bfs,s3_data=None,s6_data=None):
    V=bfs["V_pu"]; _,cumZ=_build_graph()
    VFI=np.array(s6_data["VFI"]) if s6_data else _norm01(1-V)
    ECI=np.array(s6_data["ECI"]) if s6_data else _norm01(1/(cumZ+0.01))
    instab=np.array(s6_data["instability"]) if s6_data else _norm01(1-V)
    risk=_norm01(_norm01(VFI)+_norm01(1-V)+instab)
    VSI_br=[float(4*(R*(bfs["Pl"][i]/1000)+X*(bfs["Ql"][i]/1000))**2/max(V[fb-1]**4,1e-9))
            for i,(fb,tb,R,X,*_) in enumerate(BRANCH_DATA)]
    top5=np.argsort(ECI)[::-1][:5]
    BII=np.array(s3_data["BII"]) if s3_data else risk
    DPF=np.array(s6_data.get("DPF",risk)) if s6_data else risk
    radar={"bus_labels":[f"Bus {top5[r]+1}" for r in range(5)],
           "metric_labels":["BII","VFI","DPF","ECI","Instability"],
           "profiles":{f"bus_{top5[r]+1}":[float(_norm01(BII)[top5[r]]),
                                            float(_norm01(VFI)[top5[r]]),
                                            float(_norm01(DPF)[top5[r]]),
                                            float(ECI[top5[r]]),
                                            float(instab[top5[r]])] for r in range(5)}}
    return _ser({"composite_risk_score":risk,"VSI_per_branch":VSI_br,
                 "top5_multi_index_profile":radar,"is_proxy":s3_data is None})

def compute_s10(bfs,s3_data=None,s6_data=None):
    V=bfs["V_pu"]; G,cumZ=_build_graph(); G_ug=G.to_undirected()
    for u,v,d in G_ug.edges(data=True): d["weight"]=1/(d["Z"]+1e-6)
    L=nx.laplacian_matrix(G_ug,weight="weight").toarray()
    fiedler=float(np.sort(np.real(np.linalg.eigvalsh(L)))[1])
    bc_arr=np.array([nx.betweenness_centrality(G_ug,weight="weight")[b] for b in range(1,N+1)])
    r_ZV=float(np.corrcoef(cumZ,V)[0,1])
    BII=np.array(s3_data["BII"]) if s3_data else _norm01(cumZ)*0.05
    ECI=np.array(s6_data["ECI"]) if s6_data else _norm01(1/(cumZ+0.01))
    r_ZBII=float(np.corrcoef(cumZ,BII)[0,1])
    r_bcECI=float(np.corrcoef(bc_arr,ECI)[0,1])
    VSI_br=[float(4*(R*(bfs["Pl"][i]/1000)+X*(bfs["Ql"][i]/1000))**2/max(V[fb-1]**4,1e-9))
            for i,(fb,tb,R,X,*_) in enumerate(BRANCH_DATA)]
    worst_vsi=[f"{BRANCH_DATA[k][0]}-{BRANCH_DATA[k][1]}" for k in np.argsort(VSI_br)[:3]]
    return _ser({"findings":{
        "1_weakest_buses":(np.argsort(V)[:5]+1).tolist(),
        "2_most_influential_BII":(np.argsort(BII)[::-1][:5]+1).tolist(),
        "4_highest_ECI":(np.argsort(ECI)[::-1][:5]+1).tolist(),
        "5_fiedler_lambda2":round(fiedler,5),
        "5_topology_fragility":"fragile" if fiedler<0.02 else "moderate",
        "6_r_Z_Vpu":round(r_ZV,3),"6_r_Z_BII":round(r_ZBII,3),
        "7_r_betweenness_ECI":round(r_bcECI,3),
        "8_worst_VSI_branches":worst_vsi,
        "9_feeder_end_buses":{b:{"V_pu":round(float(V[b-1]),4),
                                  "instability":round(float(_norm01(1-V)[b-1]),4)} for b in [18,25,33]},
        "10_system_summary":{"P_loss_kw":round(bfs["totPl"],2),
                              "P_loss_pct":round(bfs["totPl"]/max(bfs["p_load"].sum(),1)*100,2),
                              "Q_loss_kvar":round(bfs["totQl"],2),
                              "buses_below_095":int((V<0.95).sum()),
                              "buses_below_090":int((V<0.90).sum()),
                              "max_loading_pct":round(float(bfs["loading"].max()),1),
                              "min_V_pu":round(float(V.min()),4),
                              "min_V_bus":int(V.argmin())+1}},
        "is_proxy":s3_data is None})


# ── Background S3 ─────────────────────────────────────────────────────────────
def compute_s3_background(bfs,changes,perturbs=None):
    perturbs=perturbs or [0.05,0.10,0.20]
    base_V=bfs["V_pu"].copy()
    influ={}
    for pct in perturbs:
        M=np.zeros((N,N))
        for i in range(1,N):
            ch={"global_scale":changes.get("global_scale",1.0),
                "dg":changes.get("dg",{}),"capacitor":changes.get("capacitor",{}),
                "bus_load_scale":{**changes.get("bus_load_scale",{}),i+1:changes.get("bus_load_scale",{}).get(i+1,1.0)*(1+pct)}}
            pf=run_bfs(ch); M[i,:]=np.abs(pf["V_pu"]-base_V)
        influ[str(pct)]=M
    M20=influ.get("0.2",influ[str(perturbs[-1])])
    BII=M20.sum(axis=1); VFI=M20.sum(axis=0)
    DPF=np.array([(M20[i,:].sum()-M20[i,i])/(M20[i,i]+1e-12) for i in range(N)])
    profiles={f"bus_{b}":M20[b-1].tolist() for b in [2,8,18,24,30] if b<=N}
    return _ser({"BII":BII,"VFI":VFI,"DPF":DPF,
                 "propagation_profiles_20pct":profiles,"perturb_levels":perturbs,
                 "summary":{"most_influential_bus_BII":int(BII.argmax())+1,
                             "most_fragile_bus_VFI":int(VFI.argmax())+1,
                             "top5_BII_buses":(np.argsort(BII)[::-1][:5]+1).tolist(),
                             "top5_VFI_buses":(np.argsort(VFI)[::-1][:5]+1).tolist()}})


# ── Background S4 ─────────────────────────────────────────────────────────────
def compute_s4_background(bfs,delta_q=50.0):
    import pandapower as pp
    p_base=bfs["p_load"].copy(); q_base=bfs["q_load"].copy(); base_V=bfs["V_pu"].copy()
    dVdQ=np.zeros((N,N))
    for i in range(1,N):
        q2=q_base.copy(); q2[i]+=delta_q
        net=pp.create_empty_network(sn_mva=100)
        for j in range(N): pp.create_bus(net,vn_kv=BASE_KV)
        pp.create_ext_grid(net,bus=0,vm_pu=1.0)
        for idx2,(fb,tb,R,X,*_) in enumerate(BRANCH_DATA):
            pp.create_line_from_parameters(net,from_bus=fb-1,to_bus=tb-1,length_km=1.0,
                r_ohm_per_km=R,x_ohm_per_km=X,c_nf_per_km=0,max_i_ka=I_RATING/1000)
        for j in range(1,N): pp.create_load(net,bus=j,p_mw=p_base[j]/1000,q_mvar=q2[j]/1000)
        pp.runpp(net,algorithm="bfsw",max_iteration=200,tolerance_mva=1e-9)
        dVdQ[i,:]=(net.res_bus.vm_pu.values-base_V)/delta_q
    ss=-np.diag(dVdQ); cs=np.abs(dVdQ).sum(axis=0)
    rank_idx=np.argsort(ss)[::-1][:15]
    return _ser({"self_sensitivity":ss,"col_sum_sensitivity":cs,
                 "weak_bus_ranking":{"bus_numbers":(rank_idx+1).tolist(),"self_sens_values":ss[rank_idx].tolist()},
                 "summary":{"most_sensitive_bus":int(ss.argmax())+1,
                             "max_self_sens_pu_per_kvar":round(float(ss.max()),6),
                             "top5_weak_buses":(np.argsort(ss)[::-1][:5]+1).tolist()}})


# ── Compare ───────────────────────────────────────────────────────────────────
def compare_queries(result_a,result_b):
    def sarr(d,k):
        v=d.get("s1",d).get("after",d).get(k,[])
        return np.array(v) if v else np.zeros(N)
    Va=sarr(result_a,"V_pu"); Vb=sarr(result_b,"V_pu")
    La=sarr(result_a,"loading"); Lb=sarr(result_b,"loading")
    Pla=sarr(result_a,"Pl"); Plb=sarr(result_b,"Pl")
    return _ser({"dV_AvsB":(Va-Vb).tolist(),"dLoad_AvsB":(La-Lb).tolist(),"dPl_AvsB":(Pla-Plb).tolist(),
                 "V_a":Va.tolist(),"V_b":Vb.tolist(),
                 "loading_a":La.tolist(),"loading_b":Lb.tolist(),
                 "Pl_a":Pla.tolist(),"Pl_b":Plb.tolist(),
                 "summary":{"A_min_V":round(float(Va.min()),4),"B_min_V":round(float(Vb.min()),4),
                             "A_losses":round(float(Pla.sum()),2),"B_losses":round(float(Plb.sum()),2),
                             "A_max_load":round(float(La.max()),1),"B_max_load":round(float(Lb.max()),1)}})