import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

@dataclass
class Param:
    nOD: int
    nRS: int
    nDrv: int
    nShp: int
    T: int
    K: int
    theta: float
    phi: float

@dataclass
class Data:
    OD: np.ndarray # 2d array of (o,d) pairs for W
    RS: np.ndarray # 2d array of (r,s) pairs for J
    ship_flow: np.ndarray # number of shippers for each RS -> J x 1
    drv_flow: np.ndarray # number of drivers for each OD -> W x 1
    c_ship: np.ndarray # shipper costs -> J x (T + 1)
    c_ship_indv: np.ndarray # atomic shipper costs -> {j: Nj x (T+1)}
    c_drv: np.ndarray # driver edge costs -> W x (J + 1) x (J + 1) for od case // D x (J + 1) x (J + 1) for d case
    co: np.ndarray # driver departing edge costs for d case -> D x Od x (J + 1)
    c_drv_indv: np.ndarray # atomic driver edge costs -> {(t,w): Ntw x (J + 1) x (J + 1)} for od case // {(t,d): Ntd x (J + 1) x (J + 1)} for d case
    co_indv: np.ndarray # atomic driver departing edge costs -> {(t,d): Ntd x Od x (J + 1)}
    ship_indv_idx: np.ndarray # atomic shipper indexes -> {j: np.ndarray}
    drv_indv_idx: np.ndarray # atomic driver indexes -> {t: np.ndarray}

def setparam_(nOD: int, nRS: int, nDrv: int, nShp: int,
                T: int, K: int, theta: float, phi: float) -> Param:
    return Param(
        nOD, nRS, nDrv, nShp, T, K, theta, phi
    )

class Dataset(object):

    def __init__(self,
            link_path: str = 'data/Winnipeg/link.csv',
            od_path: str = 'data/Winnipeg/od.csv',
            cSP = None,
            od_weight = False,
            rs_weight = False,
            t_distrb = "Uniform",
            gamma = 3.0,
            speed = 55.0, # miles / hour
            vot = 15.0, # usd / hour
            base_c = 0.0
            ):
        
        # parameter
        self.prof_coef = gamma # for asking professional driver to delivery task
        self.base_c = base_c

        # weight for sampling ODs
        self.od_weight = od_weight
        self.rs_weight = rs_weight
        self.t_distrb = t_distrb

        # define network
        self.link_data = pd.read_csv(link_path)
        self.od_data = pd.read_csv(od_path)
        self.link_data['cost'] = self.link_data['free_flow_time']

        # pre-computation of shortest path costs between all possible ODs
        if cSP is not None:
            self.cSP = cSP * vot / speed
        else:
            print("Pre-computation of shortest path costs")
            G = nx.from_pandas_edgelist(self.link_data, source='init_node', target='term_node', edge_attr=['cost'], create_using=nx.DiGraph)
            origins = self.od_data['origin'].unique()
            dests = self.od_data['destination'].unique()
            zones = np.unique(np.append(origins, dests))
            Z = zones.max() + 1
            self.cSP = np.zeros((Z,Z), dtype=np.float64)
            for o in tqdm(zones):
                for d in zones:
                    self.cSP[o,d] = nx.shortest_path_length(G, source=o, target=d, weight='cost')
    
    def generate_data(self, param_: Param, N: int) -> List[Data]:
        print(f"Generate {N} data for {param_}")
        nOD = param_.nOD
        nRS = param_.nRS
        nDrv = param_.nDrv
        nShp = param_.nShp
        T = param_.T
        K = param_.K

        # sample datasets
        expData = []
        for n in tqdm(range(N)):
            # sample OD and RS
            od_weights = self.od_data.flow if self.od_weight else None
            rs_weights = self.od_data.flow if self.rs_weight else None
            od = self.od_data.sample(nOD, replace=False, weights=od_weights) # length W = nOD
            rs = self.od_data.sample(nRS, replace=False, weights=rs_weights) # length J = nRS
            OD = od[['origin', 'destination']].values
            RS = rs[['origin', 'destination']].values
            od.index = np.arange(nOD)
            rs.index = np.arange(nRS)
            dests = np.unique(OD[:,1])
            D = dests.shape[0]
            d_idxs = {d: np.where(OD[:,1] == dests[d])[0] for d in range(D)}

            # allocate drivers to OD and shippers to RS: with non-zero constraint
            while True:
                q_p = od.flow / od.flow.values.sum() if self.od_weight else None
                x_p = rs.flow / rs.flow.values.sum() if self.rs_weight else None
                qSample = np.random.choice(np.arange(nOD), nDrv, p=q_p)
                xSample = np.random.choice(np.arange(nRS), nShp, p=x_p)
                q = np.unique(qSample, return_counts=True)[1]
                x = np.unique(xSample, return_counts=True)[1]
                if (len(q.nonzero()[0]) == nOD) and (len(x.nonzero()[0]) == nRS):
                    break
            
            # allocate drivers to time windows: with non-zero constraint
            drv_flow_od = np.zeros((nOD, T), dtype=np.int64)
            p_slot = None
            if self.t_distrb == "One-Peak":
                p_slot = np.array([0.5, 0.25, 0.15, 0.1])
            elif self.t_distrb == "Two-Peak":
                p_slot = np.array([0.4, 0.1, 0.1, 0.4])
            for w in range(nOD):
                while True:
                    # random sampling
                    tSample = np.random.choice(np.arange(T), q[w], p=p_slot)
                    Nw = np.unique(tSample, return_counts=True)[1]
                    if (len(Nw.nonzero()[0]) == T):
                        break
                drv_flow_od[w] = Nw
            
            drv_flow = {}
            for d in range(D):
                od_idxs = d_idxs[d]
                drv_flow[d] = drv_flow_od[od_idxs].T # T x Od

            ## Shipper costs
            c_ship = np.zeros((nRS, T+1), dtype=np.float64)
            c_ship_indv = {}
            ship_indv_idx = {}
            cum_idx = 0
            for j in range(nRS):
                # cancel the order = ask a professional driver
                c_ship[j,-1] = self.prof_coef * self.cSP[RS[j][0], RS[j][1]]
                # base utilities for differen time slots
                c_ship += self.base_c
                # u_base_indv = 0. * np.ones((x[j], T+1), dtype=np.float64)
                # u_base_indv[:,-1] = 0.
                # pref_slot = np.random.choice(T, x[j])
                # u_base_indv[np.arange(x[j]), pref_slot] = 5.
                # modify individual cost using base utilities
                c_ship_indv[j] = c_ship[j] - np.random.gumbel(scale=1/param_.theta, size=(x[j], T+1))
                ship_indv_idx[j] = np.arange(cum_idx, cum_idx+x[j])
                cum_idx += x[j]

            ## Driver costs
            c_drv = np.zeros((D, nRS+1, nRS+1), dtype=np.float64)
            c_drv_indv = {}
            co = {}
            co_indv = {}
            # drv_indv_idx = {t: [] for t in range(T)}
            drv_indv_idx = {}
            cum_idx = 0
            td = nRS
            for d, nd in enumerate(dests):
                c_drv[d,:nRS,td] = self.cSP[RS[:,1],nd]
                c_drv[d,:nRS,:nRS] = self.cSP[RS[:,1],RS[:,0]] + self.cSP[RS[:,0],RS[:,1]]
                od_idxs = d_idxs[d]
                co[d] = np.zeros((od_idxs.shape[0],nRS+1))
                for o in range(od_idxs.shape[0]):
                    w = od_idxs[o]
                    no = OD[w,0]
                    co[d][o,:nRS] = self.cSP[no,RS[:,0]] + self.cSP[RS[:,0],RS[:,1]] - self.cSP[no,nd]
                    co[d][o,:] += self.base_c
                    for t in range(T): 
                        ctw = c_drv[d] - np.random.gumbel(scale=1/(param_.phi), size=(drv_flow_od[w,t], nRS+1, nRS+1))
                        ctw[:,nRS,:] = 0 # no cost for dummy link
                        c_drv_indv[(t,w)] = ctw
                        co_indv[(t,w)] = co[d][o] - np.random.gumbel(scale=1/(param_.phi), size=(drv_flow_od[w,t], nRS+1))
                        # store index for SOP-LP
                        # drv_indv_idx[t] = np.concatenate([drv_indv_idx[t], np.arange(cum_idx, cum_idx+drv_flow_od[w,t])])
                        drv_indv_idx[(t,w)] = np.arange(cum_idx, cum_idx+drv_flow_od[w,t])
                        cum_idx += drv_flow_od[w,t]

            expData.append(Data(
                OD = OD,
                RS = RS,
                drv_flow = drv_flow,
                ship_flow = x,
                c_ship = c_ship,
                c_ship_indv = c_ship_indv,
                c_drv = c_drv,
                co = co,
                c_drv_indv = c_drv_indv,
                co_indv = co_indv,
                ship_indv_idx = ship_indv_idx,
                drv_indv_idx = drv_indv_idx
            ))
        return expData

if __name__ == '__main__':
    
    np.random.seed(222)

    nOD = 5
    nRS = 10
    nDrv = 1500
    nShp = 4500
    T = 4
    K = 3
    theta = 1.
    phi = 1.

    param_ = setparam_(nOD, nRS, nDrv, nShp, T, K, theta, phi)
    print(param_)

    cSP_load = np.load('data/Winnipeg/cSP.npy')
    dataset = Dataset(cSP=cSP_load)
    expData = dataset.generate_data(param_, N=1)
    # print(expData[0].drv_flow.shape)
    # print(expData[0].ship_flow.shape)
    # print(expData[0].c_ship.shape)
    # print(expData[0].c_ship_indv[0].shape)
    # print(expData[0].c_drv_indv[(0,0)].shape)

    from network import MNL_od, MNL_d
    from model import CSD
    
    # loader = MNL_od(param_, expData[0])
    # p = np.ones((T,nRS), dtype=np.float64) * 7.5
    # y, S_ship = loader.compute_y(p)
    # x, S_drv = loader.compute_x(p)
    # print(y.shape)
    # print(x.shape)
    # # print((y[:,:T].T - x.sum(axis=1)).sum())
    # print(expData[0].drv_flow[0,1])

    csd = CSD(MNL_od)
    csd.load_data(param_, expData)
    resLP, solLP = csd.solve_SOP_lp()
    print(resLP)
    # print(solLP)
    # resAGD, FPsol = csd.solve_AGD(r=0, g0=1e-4, with_BT=True)
    # print(resAGD)
    resFP, solFP = csd.solve_fluid_particle()
    metrics = csd.compare_results(resLP, solLP, resFP, solFP)
    metrics = pd.DataFrame(metrics)
    print(metrics)