from dataset import *
from network import *
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
import pulp as pl
from tqdm import tqdm
from utils import Timer
import math
timer = Timer()

def get_stats(d: np.ndarray, dName: str = 'd') -> dict:
    return {
        f'sum_{dName}': np.sum(d),
        f'mean_{dName}': np.mean(d),
        f'max_{dName}': np.max(d),
        f'min_{dName}': np.min(d),
        f'std_{dName}': np.std(d),
        f'nonneg_{dName}': (d >= 0).sum(),
    }

def my_round(x, decimals=0):
    return np.floor(x * 10**decimals + 0.5) / 10**decimals

class CSD(object):

    def __init__(self,
            nw_model = MNL_od,
            solver: str = 'gurobi', nThrd: int = 1, msg: bool = False,
            init_step_size: float = 1e-3, min_step_size: float = 1e-6, 
            accuracy: float = 1e-2, backIter: int = 500,
            maxIter: int = 1000, solve_dual: bool = False):
        self.eps = 1e-8
        self.f_lb = 5
        self.g0 = init_step_size
        self.gmin = min_step_size
        self.tol = accuracy
        self.maxIter = maxIter
        self.m_back = backIter
        self.solver = {
            'cmc': pl.PULP_CBC_CMD(threads = nThrd, msg = msg),
            'gurobi': pl.GUROBI(threads = nThrd, msg = msg),
            'cplex': pl.CPLEX_CMD(threads = nThrd, msg = msg),
        }.get(solver)
        self.solve_dual = solve_dual
        self.nw_model = nw_model

    def load_data(self, param_: Param, expData: List[Data]):
        self.W = param_.nOD
        self.J = param_.nRS
        self.A = param_.nDrv
        self.B = param_.nShp
        self.T = param_.T
        self.K = param_.K
        self.theta = param_.theta
        self.phi = param_.phi
        self.nExp = len(expData)
        self.data = expData
        self.loaders = [self.nw_model(param_, dataset) for dataset in expData]
    
    def solve_AGD(self,
                  r: int,  
                  g0: float = 1e-5, 
                  nu: float = 0.95,
                  l_min: int = 50,
                  with_BT: bool = True) -> Tuple[List[dict], List[dict]]:
        
        print(f"Solving master problem for data {r}...")

        # loader
        loader = self.loaders[r]

        # initialization
        p = {0: np.ones((self.T, self.J), dtype=np.float64)*10}
        p_aux = np.zeros((self.T, self.J), dtype=np.float64)
        m = 0 # iteration counter
        l = 0 # adaptive restart
        t = {0: 1.} # momentum
        x = {}
        y = {}
        obj_val = {}
        gamma = {0: g0}
        
        # adjust
        first_trial = True
        mBT_coef = 1        

        # start AGD algorithm
        timer.start()
        while True:
            # network loading
            y[m], S_ship = loader.compute_y(p[m])
            x_edge, x[m], S_drv = loader.compute_x(p[m])
            grad = y[m][:self.T,:] - x[m].sum(axis=1) # T x J
            obj_val[m] = S_ship.sum() + S_drv.sum()

            ## skip the backtracking step due to inefficiency
            if with_BT and m > mBT_coef*self.m_back and gamma[m] > self.gmin:
                i = 0
                while True:
                    gamma[m+1] = (nu**i) * gamma[m]
                    p_new = p_aux + gamma[m+1] * grad
                    p_new = np.clip(p_new, 0., None)
                    _, S_ship = loader.compute_y(p_new)
                    S_drv = loader.compute_S_drv(p_new)
                    p_dif = p_new - p_aux
                    F = S_ship.sum() + S_drv.sum()
                    Q = obj_val[m] + np.linalg.norm(p_dif * grad) + np.linalg.norm(p_dif)/(gamma[m+1] * 2)
                    # Q = obj_val[m] + np.linalg.norm(grad) * gamma[m+1]/2
                    if F >= Q or i >= 5:
                        # if i > 0:
                        #     print(f"Backtracking -- gamma={gamma[m+1]}, i={i}")
                        break
                    else:
                        i += 1
            else:
                gamma[m+1] = gamma[m]
                p_new = p_aux + gamma[m+1] * grad

            # updating
            # p_new = p_aux + gamma[m+1] * grad
            p[m+1] = p_new
            t[l+1] = (1 + math.sqrt(1 + 4 * (t[l]**2)))/2
            p_aux = p[m+1] + ((t[l]-1)/t[l+1]) * (p[m+1] - p[m])
            p_aux = np.clip(p_aux, 0., None)

            # adaptive restart: function
            if m > 0 and l > l_min:
                # if np.sum(grad * (p[m+1] - p[m])) < 0: # the same as obj
                if obj_val[m] < obj_val[m-1]:
                    print(f'Adaptive restart: m={m}, l={l}')
                    t = {0: 1.}
                    l = 0
                else:
                    l += 1
            else:
                l += 1

            # convergence test
            if m == 0: 
                m += 1
                continue
            dif_p = np.abs(p[m+1] - p[m])/np.clip(p[m+1], 1., None)
            dif_G = np.abs((obj_val[m] - obj_val[m-1])/obj_val[m-1])
            max_gap = (np.abs(grad)).max() # np.abs(grad).max()
            if m % 100 == 0:
                print(m, p[m].mean(), obj_val[m], max_gap, t[l], gamma[m]) #, np.sum(C_drv), np.sum(H_drv)
            if m > self.maxIter or (np.max(dif_p) < 1e-4 and dif_G < 1e-6 and max_gap < self.tol):
                if max_gap > 10 and first_trial:
                    first_trial = False
                    mBT_coef *= 0.5
                    # initialization
                    p = {0: np.ones((self.T, self.J), dtype=np.float64)*10}
                    p_aux = np.zeros((self.T, self.J), dtype=np.float64)
                    m = 0 # iteration counter
                    l = 0 # adaptive restart
                    t = {0: 1.} # momentum
                    x = {}
                    y = {}
                    obj_val = {}
                    gamma = {0: g0*10}
                    # restart
                    print('Restart as AGD does not satisfy criterion.')
                    timer.start()
                    continue
                else:                   
                    print(f'AGD converged at m={m} with dif_p={np.mean(dif_p)} and dif_G={dif_G} and grad={max_gap}')
                    break
            m += 1
        
        # end AGD algorithm
        runtime = timer.stop()
            
        record = {
            "data": r,
            "opt_val": obj_val[m],
            "cpu_time": runtime,
            "max_rel_dif_p": dif_p.max(),
            "rel_dif_obj": dif_G,
            "max_grad": max_gap,
        }
        solution = {
            "y": y[m],
            "q": x[m],
            "p": p[m],
            "grad": grad,
            "step_size": gamma[m],
            "nIter": m,
            "x": x_edge,
        }
        
        return record, solution

    def solve_SOP_lp(self) -> Tuple[List[dict], List[dict]]:
        """
        Naive method to solve [SO-P]

        Returns:
            Tuple[np.ndarray, float]: optimal solution and optimal value.
        """
        print("Solving matching problem as naive LP")

        records, solutions = [], []
        for r in range(self.nExp):
            # input
            data_ = self.data[r]
            drv_flow = data_.drv_flow
            ship_flow = data_.ship_flow
            ship_indv_idx = data_.ship_indv_idx # shipper indexes for each j
            drv_indv_idx = data_.drv_indv_idx # driver indexes for each t
            c_ship_indv = data_.c_ship_indv # {j: |Bj| x |T+1|}
            c_drv_indv = data_.c_drv_indv # {(t,w): |At,w| x |J+1| x |J+1|}
            co_indv = data_.co_indv # {(t,w): |At,w| x |J+1|}
            c_ship = list(c_ship_indv.values())
            c_drv = list(c_drv_indv.values())
            co = list(co_indv.values())
            c_ship = np.concatenate(c_ship, axis=0) # |B| x |T+1|
            c_drv = np.concatenate(c_drv, axis=0) # |A| x |J+1| x |J+1|
            co = np.concatenate(co, axis=0) # |A| x |J+1|

            # timer start
            timer.start()

            # define model
            model = pl.LpProblem(name="matching", sense=pl.LpMinimize)
            # define decision variables
            n = np.array(pl.LpVariable.matrix('n', (range(self.B), range(self.T + 1)), lowBound=0, upBound=1))
            m = np.array(pl.LpVariable.matrix('m', (range(self.K), range(self.A), range(self.J + 1), range(self.J + 1)), lowBound=0, upBound=1))
            m0 = np.array(pl.LpVariable.matrix('m0', (range(self.A), range(self.J + 1)), lowBound=0, upBound=1))
            # define objective
            model += pl.lpSum(c_ship * n) + pl.lpSum(c_drv * m) + pl.lpSum(co * m0), 'Objective'
            # set constraints
            for a in range(self.A):
                model += pl.lpSum(m0[a,:]) == 1, f'DrvDep{a}'
                model += pl.lpSum(m[self.K-1,a,:,self.J]) == 1, f'DrvArr{a}'
                model += pl.lpSum(m[:,a,self.J,:self.J]) == 0, f'DrvAfterArr{a}'
                for i in range(self.J+1):
                    model += m0[a,i] - pl.lpSum(m[0,a,i,:]) == 0, f'DrvConsv_{a}_0_{i}'
                    for k in range(1,self.K):
                        # model += pl.lpSum(m[k-1,a,:,self.J]) - m[k,a,self.J,self.J] == 0, f'DrvConsv_{a}_{k}_d'
                        model += pl.lpSum(m[k-1,a,:,i]) - pl.lpSum(m[k,a,i,:]) == 0, f'DrvConsv_{a}_{k}_{i}'
            for b in range(self.B):
                model += pl.lpSum(n[b,:]) == 1, f'Ship{b}'
            for t in range(self.T):
                # drv_idx = drv_indv_idx[t].astype(np.int64)
                idxs = [drv_indv_idx[(t,w)] for w in range(self.W)]
                drv_idx = np.concatenate(idxs).astype(np.int64)
                for j in range(self.J):
                    ship_idx = ship_indv_idx[j].astype(np.int64)
                    model += pl.lpSum(n[ship_idx,t]) <= pl.lpSum(m[:,drv_idx,:,j]) + pl.lpSum(m0[drv_idx,j]), f'SupDem_{t}_{j}'
            
            # solve LP
            result = model.solve(self.solver)
            runtime = timer.stop()
            nArray = np.array([n[b,t].varValue for b in range(self.B) for t in range(self.T+1)]).reshape(self.B, self.T+1)
            mArray = np.array([m[k,a,i,j].varValue for k in range(self.K) for a in range(self.A) for i in range(self.J+1) for j in range(self.J+1)]).reshape(self.K, self.A, self.J+1, self.J+1)
            m0Array = np.array([m0[a,j].varValue  for a in range(self.A) for j in range(self.J+1)]).reshape(self.A, self.J+1)
            # shipper choice           
            zopt = pl.value(model.objective)
            y = np.zeros((self.T+1, self.J), dtype=np.float64)
            z_ship = np.zeros((self.J,), dtype=np.float64)
            for j, Nj in enumerate(ship_flow):
                ship_idx = ship_indv_idx[j].astype(np.int64)
                nj = nArray[ship_idx]
                y[:,j] = nj.sum(axis=0)
                z_ship[j] = np.sum(nj * c_ship_indv[j])
            x = np.zeros((self.T, self.W, self.J), dtype=np.float64)
            z_drv = np.zeros((self.T, self.W), dtype=np.float64)
            for (t,w), co_tw in co_indv.items():
                idxs = drv_indv_idx[(t,w)]
                mtw = mArray[:,idxs,:,:]
                m0tw = m0Array[idxs,:]
                x[t,w] = mtw.sum(axis=(0,1,2))[:self.J] + m0tw.sum(axis=0)[:self.J]
                z_drv[t,w] = np.sum(mtw * c_drv_indv[(t,w)]) + np.sum(m0tw * co_tw)

            # Lagrangian multipliers
            p = np.array([model.constraints[f'SupDem_{t}_{j}'].pi for t in range(self.T) for j in range(self.J)]).reshape(self.T,self.J)
            records.append({
                "data": r,
                "status": pl.LpStatus[result],
                "opt_val": zopt,
                "cpu_time": runtime,
                "ship_rate_N1": 1. - (y[self.T,:].sum()/y.sum())
            })
            solutions.append({
                "y": y,
                "x": x,
                "p": -p,
                "z_drv": z_drv.sum(),
                "z_ship": z_ship.sum(),
                "z_drv_od": z_drv,
                "z_ship_j": z_ship,
            })
            print(f"Problem solved for data {r}. Zopt: {zopt:.3f}; with {runtime:.3f}s")

        return records, solutions

    def solve_SOP_lp_N0(self) -> Tuple[List[dict], List[dict]]:
        """
        Naive method to solve [SO-P] without private info

        Returns:
            Tuple[np.ndarray, float]: optimal solution and optimal value.
        """
        print("Solving matching problem as naive LP")

        records, solutions = [], []
        for r in range(self.nExp):
            # input
            data_ = self.data[r]
            OD = data_.OD
            dests = np.unique(OD[:,1])
            D = dests.shape[0]
            d_idxs = {d: np.where(OD[:,1] == dests[d])[0] for d in range(D)}
            drv_flow = data_.drv_flow
            ship_flow = data_.ship_flow
            ship_indv_idx = data_.ship_indv_idx # shipper indexes for each j
            drv_indv_idx = data_.drv_indv_idx # driver indexes for each t
            c_ship = data_.c_ship
            c_drv = data_.c_drv
            co = data_.co
            c_ship_indv = data_.c_ship_indv # {j: |Bj| x |T+1|}
            c_drv_indv = data_.c_drv_indv # {(t,w): |At,w| x |J+1| x |J+1|}
            co_indv = data_.co_indv # {(t,w): |At,w| x |J+1|}
            c_ship_flat, c_drv_flat, co_flat = {}, {}, {}
            # update shipper perceived cost to deterministic
            for j, cs in c_ship_indv.items():
                c_ship_flat[j] = np.tile(c_ship[j,:], (cs.shape[0],1))
            # update driver perceived cost to deterministic
            for d in range(dests.shape[0]):
                od_idxs = d_idxs[d]
                for o in range(od_idxs.shape[0]):
                    w = od_idxs[o]
                    for t in range(self.T):
                        c_drv_flat[(t,w)] = np.tile(c_drv[d,:,:], (c_drv_indv[(t,w)].shape[0],1,1))
                        co_flat[(t,w)] = np.tile(co[d][o], (c_drv_indv[(t,w)].shape[0],1))
            c_ship_flat = list(c_ship_flat.values())
            c_drv_flat = list(c_drv_flat.values())
            co_flat = list(co_flat.values())
            c_ship_flat = np.concatenate(c_ship_flat, axis=0) # |B| x |T+1|
            c_drv_flat = np.concatenate(c_drv_flat, axis=0) # |A| x |J+1| x |J+1|
            co_flat = np.concatenate(co_flat, axis=0) # |A| x |J+1|
            
            # timer start
            timer.start()
            
            # define model
            model = pl.LpProblem(name="matching", sense=pl.LpMinimize)
            # define decision variables
            n = np.array(pl.LpVariable.matrix('n', (range(self.B), range(self.T + 1)), lowBound=0, upBound=1))
            m = np.array(pl.LpVariable.matrix('m', (range(self.K), range(self.A), range(self.J + 1), range(self.J + 1)), lowBound=0, upBound=1))
            m0 = np.array(pl.LpVariable.matrix('m0', (range(self.A), range(self.J + 1)), lowBound=0, upBound=1))
            # define objective
            model += pl.lpSum(c_ship_flat * n) + pl.lpSum(c_drv_flat * m) + pl.lpSum(co_flat * m0), 'Objective'
            # set constraints
            for a in range(self.A):
                model += pl.lpSum(m0[a,:]) == 1, f'DrvDep{a}'
                model += pl.lpSum(m[self.K-1,a,:,self.J]) == 1, f'DrvArr{a}'
                model += pl.lpSum(m[:,a,self.J,:self.J]) == 0, f'DrvAfterArr{a}'
                for i in range(self.J+1):
                    model += m0[a,i] - pl.lpSum(m[0,a,i,:]) == 0, f'DrvConsv_{a}_0_{i}'
                    for k in range(1,self.K):
                        model += pl.lpSum(m[k-1,a,:,i]) - pl.lpSum(m[k,a,i,:]) == 0, f'DrvConsv_{a}_{k}_{i}'
            for b in range(self.B):
                model += pl.lpSum(n[b,:]) == 1, f'Ship{b}'
            for t in range(self.T):
                idxs = [drv_indv_idx[(t,w)] for w in range(self.W)]
                drv_idx = np.concatenate(idxs).astype(np.int64)
                for j in range(self.J):
                    ship_idx = ship_indv_idx[j].astype(np.int64)
                    model += pl.lpSum(n[ship_idx,t]) <= pl.lpSum(m[:,drv_idx,:,j]) + pl.lpSum(m0[drv_idx,j]), f'SupDem_{t}_{j}'
            
            # solve LP
            result = model.solve(self.solver)
            runtime = timer.stop()
            nArray = np.array([n[b,t].varValue for b in range(self.B) for t in range(self.T+1)]).reshape(self.B, self.T+1)
            mArray = np.array([m[k,a,i,j].varValue for k in range(self.K) for a in range(self.A) for i in range(self.J+1) for j in range(self.J+1)]).reshape(self.K, self.A, self.J+1, self.J+1)
            m0Array = np.array([m0[a,j].varValue  for a in range(self.A) for j in range(self.J+1)]).reshape(self.A, self.J+1)
            # shipper choice           
            zopt = pl.value(model.objective)
            y = np.zeros((self.T+1, self.J), dtype=np.float64)
            z_ship = np.zeros((self.J,), dtype=np.float64)
            for j, Nj in enumerate(ship_flow):
                ship_idx = ship_indv_idx[j].astype(np.int64)
                nj = nArray[ship_idx]
                y[:,j] = nj.sum(axis=0)
                z_ship[j] = np.sum(nj * c_ship_indv[j])
            x = np.zeros((self.T, self.W, self.J), dtype=np.float64)
            z_drv = np.zeros((self.T, self.W), dtype=np.float64)
            for (t,w), co_tw in co_indv.items():
                idxs = drv_indv_idx[(t,w)]
                mtw = mArray[:,idxs,:,:]
                m0tw = m0Array[idxs,:]
                x[t,w] = mtw.sum(axis=(0,1,2))[:self.J] + m0tw.sum(axis=0)[:self.J]
                z_drv[t,w] = np.sum(mtw * c_drv_indv[(t,w)]) + np.sum(m0tw * co_tw)

            # Lagrangian multipliers
            p = np.array([model.constraints[f'SupDem_{t}_{j}'].pi for t in range(self.T) for j in range(self.J)]).reshape(self.T,self.J)
            records.append({
                "data": r,
                "status": pl.LpStatus[result],
                "opt_val": zopt,
                "cpu_time": runtime,
                "ship_rate_N0": 1. - (y[self.T,:].sum()/y.sum())
            })
            solutions.append({
                "y": y,
                "x": x,
                "p": -p,
                "z_drv": z_drv.sum(),
                "z_ship": z_ship.sum(),
                "z_drv_od": z_drv,
                "z_ship_j": z_ship,
            })
            print(f"Problem solved for data {r}. Zopt: {zopt:.3f}; with {runtime:.3f}s")

        return records, solutions
    
    def solve_SO_Sub_ship(self, j: int, c: np.ndarray, y: np.ndarray) -> dict:
        """
        Solve [SO/Sub(od)] by LP solver

        Args:
            j (int): id of task
            c (np.ndarray): atomic cost matrix |B_j| x |T+1|
            y (np.ndarray): vector of no. pertmits |T+1| x 1

        Returns:
            dict: optimal solution
        """
        N = c.shape[0]

        # timer start
        timer.start()
            
        # define LP model
        model = pl.LpProblem(name="matching", sense=pl.LpMinimize)
        # define decision variables
        n = np.array(pl.LpVariable.matrix('n', (range(N), range(self.T + 1)), lowBound=0, upBound=1))
        # define objective
        model += pl.lpSum(c * n), 'Objective'
        # set constraints
        for b in range(N):
            model += pl.lpSum(n[b,:]) == 1, f'Ship{b}'
        for t in range(self.T):
            model += pl.lpSum(n[:,t]) == y[t], f'Cap{t}'
        
        # solve LP
        result = model.solve(self.solver)
        runtime = timer.stop()
        # objective
        zopt = pl.value(model.objective)
        # solution
        nArray = np.array([n[b,t].varValue for t in range(self.T+1) for b in range(N)]).reshape(N, self.T+1)
        # reward
        p = np.array([model.constraints[f'Cap{t}'].pi for t in range(self.T)])
        return {
            "task": j,
            "status": pl.LpStatus[result],
            "cpu_time": runtime,
            "opt_val": zopt, 
            "n": nArray,
            "p": p,
        }
    
    def solve_SO_Sub_drv_nonzero(self, t: int, w: int, c: np.ndarray, co: np.ndarray, x: np.ndarray, xo: np.ndarray) -> dict:
        """
        Solve [SO/Sub(od)] by LP solver
        Remove tasks with zero allocation

        Args:
            t (int): id of time window
            w (int): id of OD pair
            c (np.ndarray): atomic cost matrix |A_(t,w))| x |J+1| x |J+1|
            co (np.ndarray): atomic cost for departing edge |A_(t,w))| x |J+1|
            x (np.ndarray): edge flow |K| x |J| x |J|
            xo (np.ndarray): flow for departing edge |J| x 1
            # q (np.ndarray): vector of no. tasks |J| x 1

        Returns:
            dict: optimal solution
        """
        
        if xo.sum() < 1:
            return {
                "timw": t,
                "od": w,
                "status": "",
                "cpu_time": 0.,
                "opt_val": co[:,self.J].sum(), # everyone use link o-d
                "m": None,
                "m0": None,
                "p": None,
            }
        else:
            # nonzero_idxs = np.where(q > 0)[0]
            nonzero_idxs = np.where(xo > 0)[0]
            # print(xo, xo.shape, nonzero_idxs)
            J = nonzero_idxs.shape[0]
            reduced_idxs = np.append(nonzero_idxs, self.J)
            c_reduced = c[:,reduced_idxs,:][:,:,reduced_idxs]
            co_reduced = co[:,reduced_idxs]
        
        N = c.shape[0]

        # timer start
        timer.start()
            
        # define LP model
        model = pl.LpProblem(name="matching", sense=pl.LpMinimize)
        # define decision variables
        m = np.array(pl.LpVariable.matrix('m', (range(self.K), range(N), range(J+1), range(J+1)), lowBound=0, upBound=1))
        m0 = np.array(pl.LpVariable.matrix('m0', (range(N), range(J+1)), lowBound=0, upBound=1))
        # define objective
        model += pl.lpSum(c_reduced * m) + pl.lpSum(co_reduced * m0), 'Objective'
        # set constraints
        # flow conservation
        for a in range(N):
            model += pl.lpSum(m0[a,:]) == 1, f'DrvDep{a}'
            model += pl.lpSum(m[self.K-1,a,:,J]) == 1, f'DrvArr{a}'
            model += pl.lpSum(m[:,a,J,:J]) == 0, f'DrvAfterArr{a}'
            for i in range(J+1):
                model += m0[a,i] - pl.lpSum(m[0,a,i,:]) == 0, f'DrvConsv_{a}_0_{i}'
                for k in range(1,self.K):
                    model += pl.lpSum(m[k-1,a,:,i]) - pl.lpSum(m[k,a,i,:]) == 0, f'DrvConsv_{a}_{k}_{i}'
        
        for j, tj in enumerate(nonzero_idxs):
            model += pl.lpSum(m0[:,j]) == xo[tj], f'Cap_dep_{j}'
            for i, ti in enumerate(nonzero_idxs):
                for k in range(self.K):
                    model += pl.lpSum(m[k,:,i,j]) == x[k,ti,tj], f'Cap_{k}_{i}_{j}'
        
        # solve LP
        result = model.solve(self.solver)
        runtime = timer.stop()
        # objective
        zopt = pl.value(model.objective)
        # solution
        mArray = np.array([m[k,a,i,j].varValue for k in range(self.K) for a in range(N) for i in range(J+1) for j in range(J+1)]).reshape(self.K, N, J+1, J+1)
        m0Array = np.array([m0[a,j].varValue for a in range(N) for j in range(J+1)]).reshape(N, J+1)
        
        # revert to original size
        m_orig = np.zeros((self.K,N,self.J+1,self.J+1), dtype=np.float64)
        m0_orig = np.zeros((N,self.J+1), dtype=np.float64)
        p = np.zeros((self.J,), dtype=np.float64)
        for i, ti in enumerate(reduced_idxs):
            m_orig[:,:,ti,reduced_idxs] = mArray[:,:,i,:]
            m0_orig[:,ti] = m0Array[:,i]

        return {
            "timw": t,
            "od": w,
            "status": pl.LpStatus[result],
            "cpu_time": runtime,
            "opt_val": zopt, 
            "m": m_orig,
            "m0": m0_orig,
            "p": p,
        }
    
    def solve_SO_Sub_drv_nonzero_q(self, t: int, w: int, c: np.ndarray, co: np.ndarray, q: np.ndarray) -> dict:
        """
        Solve [SO/Sub(od)] by LP solver
        Remove tasks with zero allocation

        Args:
            t (int): id of time window
            w (int): id of OD pair
            c (np.ndarray): atomic cost matrix |A_(t,w))| x |J+1| x |J+1|
            co (np.ndarray): atomic cost for departing edge |A_(t,w))| x |J+1|
            q (np.ndarray): vector of no. tasks |J| x 1

        Returns:
            dict: optimal solution
        """
        
        if q.sum() < 1:
            return {
                "timw": t,
                "od": w,
                "status": "",
                "cpu_time": 0.,
                "opt_val": co[:,self.J].sum(), # everyone use link o-d
                "m": None,
                "m0": None,
                "p": None,
            }
        else:
            nonzero_idxs = np.where(q > 0)[0]
            J = nonzero_idxs.shape[0]
            reduced_idxs = np.append(nonzero_idxs, self.J)
            c_reduced = c[:,reduced_idxs,:][:,:,reduced_idxs]
            co_reduced = co[:,reduced_idxs]
        
        N = c.shape[0]

        # timer start
        timer.start()
            
        # define LP model
        model = pl.LpProblem(name="matching", sense=pl.LpMinimize)
        # define decision variables
        m = np.array(pl.LpVariable.matrix('m', (range(self.K), range(N), range(J+1), range(J+1)), lowBound=0, upBound=1))
        m0 = np.array(pl.LpVariable.matrix('m0', (range(N), range(J+1)), lowBound=0, upBound=1))
        # define objective
        model += pl.lpSum(c_reduced * m) + pl.lpSum(co_reduced * m0), 'Objective'
        # set constraints
        # flow conservation
        for a in range(N):
            model += pl.lpSum(m0[a,:]) == 1, f'DrvDep{a}'
            model += pl.lpSum(m[self.K-1,a,:,J]) == 1, f'DrvArr{a}'
            model += pl.lpSum(m[:,a,J,:J]) == 0, f'DrvAfterArr{a}'
            for i in range(J+1):
                model += m0[a,i] - pl.lpSum(m[0,a,i,:]) == 0, f'DrvConsv_{a}_0_{i}'
                for k in range(1,self.K):
                    model += pl.lpSum(m[k-1,a,:,i]) - pl.lpSum(m[k,a,i,:]) == 0, f'DrvConsv_{a}_{k}_{i}'
        for j, tj in enumerate(nonzero_idxs):
            model += pl.lpSum(m0[:,j]) + pl.lpSum(m[:,:,:,j]) == q[tj], f'Cap_{j}'
        
        # solve LP
        result = model.solve(self.solver)
        runtime = timer.stop()
        # objective
        zopt = pl.value(model.objective)
        # solution
        mArray = np.array([m[k,a,i,j].varValue for k in range(self.K) for a in range(N) for i in range(J+1) for j in range(J+1)]).reshape(self.K, N, J+1, J+1)
        m0Array = np.array([m0[a,j].varValue for a in range(N) for j in range(J+1)]).reshape(N, J+1)
        
        # revert to original size
        m_orig = np.zeros((self.K,N,self.J+1,self.J+1), dtype=np.float64)
        m0_orig = np.zeros((N,self.J+1), dtype=np.float64)
        p = np.zeros((self.J,), dtype=np.float64)
        # p[nonzero_idxs] = p_reduced
        for i, ti in enumerate(reduced_idxs):
            m_orig[:,:,ti,reduced_idxs] = mArray[:,:,i,:]
            m0_orig[:,ti] = m0Array[:,i]
        # p[nonzero_idxs] = p_reduced

        return {
            "timw": t,
            "od": w,
            "status": pl.LpStatus[result],
            "cpu_time": runtime,
            "opt_val": zopt, 
            "m": m_orig,
            "m0": m0_orig,
            "p": p,
        }

    def solve_fluid_particle(self, vcg: bool = False) -> Tuple[List[dict], List[dict]]:
        """
        Fluid particle decomposition approach to solve [SO-P]
            - Master Problem [SO-A-P] solved by Bregman's balancing method
            - Sub Problem [SO-Sub/(od)] solved by LP solver

        Returns:
            Tuple[np.ndarray, float]: optimal solution and optimal value.
        """
        print("Solving matching problem with fluid-particle decomposition approach")
        records, solutions = [], []
        for r in range(self.nExp):
            data_ = self.data[r]
            drv_flow = data_.drv_flow
            ship_flow = data_.ship_flow
            c_ship_indv = data_.c_ship_indv # {j: |Bj| x |T+1|}
            c_drv_indv = data_.c_drv_indv # {(t,w): |At,w| x |J+1| x |J+1|}
            co_indv = data_.co_indv # {(t,w): |At,w| x |J+1|}

            # read data
            self.OD = data_.OD
            self.RS = data_.RS
            self.dests = np.unique(self.OD[:,1])
            self.D = self.dests.shape[0]
            self.d_idxs = {d: np.where(self.OD[:,1] == self.dests[d])[0] for d in range(self.D)}

            # master problem
            FPres, FPsol = self.solve_AGD(r, g0=self.g0, with_BT=True)
            print(f"Master obj.: {FPres['opt_val']:.3f}")
            print(f"Master cpu-time: {FPres['cpu_time']:.3f}s")

            # task partition variables
            y = FPsol['y']
            x = FPsol['x']
            q = FPsol['q']

            # shipper sub-problems
            z_sub_ship, z_sub_drv = 0, 0
            runtimes_ship, runtimes_drv = [], []
            z_ships = np.zeros((self.J,), dtype=np.float64)
            z_drvs = np.zeros((self.T, self.W), dtype=np.float64)
            for j in range(self.J):
                yj = np.round(y[:,j]).astype(np.int64)
                yj = self.check_flow_consv(yj, None, ship_flow[j])
                print(j, yj, ship_flow[j])
                sol_j = self.solve_SO_Sub_ship(j, c_ship_indv[j], yj)
                z_ships[j] = sol_j['opt_val']
                z_sub_ship += sol_j['opt_val']
                runtimes_ship.append(sol_j['cpu_time'])
            cpu_time_ship = np.mean(runtimes_ship)

            for (t, w, o, d) in self.loaders[r].keys_:
                qtw = np.round(q[t,w,:]).astype(np.int64)
                xo_tw, x_tw = x[(t,w)]
                xo_tw = np.round(xo_tw).astype(np.int64)
                x_tw = np.round(x_tw).astype(np.int64)
                print(t, w, qtw, drv_flow[d][t,o], xo_tw[-1])
                sol_tw = self.solve_SO_Sub_drv_nonzero_q(t, w, c_drv_indv[(t,w)], co_indv[(t,w)], qtw)
                if sol_tw['opt_val'] is None:
                    print("Flow conservation and capacity constraint are not satisfied simultaneously.")
                    xo_tw, x_tw = self.check_flow_consv(xo_tw, x_tw, drv_flow[d][t,o])
                    # print(t, w, xo_tw, drv_flow[d][t,o])
                    sol_tw = self.solve_SO_Sub_drv_nonzero(t, w, c_drv_indv[(t,w)], co_indv[(t,w)], x_tw[:self.K, :self.J, :self.J], xo_tw[:self.J])
                z_drvs[t,w] = sol_tw['opt_val']
                z_sub_drv += sol_tw['opt_val']
                if qtw.sum() > 1:
                    runtimes_drv.append(sol_tw['cpu_time'])
            cpu_time_drv = np.mean(runtimes_drv)
        
            records.append({
                "data": r,
                "opt_val": z_sub_drv+z_sub_ship,
                "cpu_time_FP": FPres['cpu_time']+cpu_time_drv+cpu_time_ship,
                "cpu_time_master": FPres['cpu_time'],
                "cpu_time_Sub_drv": cpu_time_drv,
                "cpu_time_Sub_ship": cpu_time_ship,
                "ship_rate_FP": 1. - (y[self.T,:].sum()/y.sum())
            })
            solutions.append({
                "y": y, 
                "x": x, 
                "q": q, 
                "p": FPsol['p'],
                "z_drv": z_sub_drv,
                "z_ship": z_sub_ship,
                "z_drv_od": z_drvs,
                "z_ship_j": z_ships
            })
            print(f"Sub obj.: total={z_sub_drv+z_sub_ship:.3f}, ship={z_sub_ship:.3f}, drv={z_sub_drv:.3f}.")
            print(f"Sub cpu-time: total={cpu_time_drv+cpu_time_ship:.3f}s, ship={cpu_time_ship:.3f}s, drv={cpu_time_drv:.3f}s.")

        return records, solutions
    
    def check_flow_consv(self, xo: np.ndarray, x: np.ndarray, flow: int):
        """
        Check if rounded flows satisfy conservation laws & modify flows if not
        """
        # check xo
        if xo.sum() > flow:
            gap = xo.sum() - flow
            idxs_ = np.arange(xo.shape[0])
            np.random.shuffle(idxs_) # to avoid always modifying the values with small indexes 
            for i in idxs_:
                if xo[i] > 0:
                    xo[i] -= 1
                    gap -= 1
                if gap == 0:
                    break
        elif xo.sum() < flow:
            gap = flow - xo.sum()
            idxs_ = np.arange(xo.shape[0])
            np.random.shuffle(idxs_) # to avoid always modifying the values with small indexes 
            for i in idxs_:
                if xo[i] > 0:
                    xo[i] += 1
                    gap -= 1
                if gap == 0:
                    break
        
        if x is None:
            return xo
        
        # check x
        z = xo
        for k in range(self.K):
            for j in range(self.J+1):
                if x[k,j,:].sum() == z[j]:
                    # this is OK
                    continue
                gap = np.abs(x[k,j,:].sum() - z[j])
                pos = x[k,j,:].sum() - z[j] > 0
                print(f"gap: {gap} at {k}, {j}")
                while True:
                    if x[k,j,:].sum() == 0:
                        x[k,j,self.J] = 1
                        gap -= 1
                    else:
                        idxs_ = np.arange(self.J+1)
                        np.random.shuffle(idxs_) # to avoid always modifying the values with small indexes 
                        for i in idxs_:
                            if x[k,j,i] > 0:
                                x[k,j,i] = x[k,j,i] - 1 if pos else x[k,j,i] + 1
                                gap -= 1
                            if gap == 0:
                                break
                    if gap == 0:
                        break
            z = x[k,:,:].sum(axis=0)
        return xo, x

    def compare_results(self,
            recLP: List[dict], solLP: List[dict],
            recFP: List[dict], solFP: List[dict],
            model_name: str = "FP",
            ) -> List[dict]:

        metrics = []
        for n in range(self.nExp):
            # relative error of objective values
            zSO = recLP[n]["opt_val"]
            zFP = recFP[n]["opt_val"]
            z_dif = np.abs((zSO - zFP) / zSO)
            metric = {f"z_dif_{model_name}": z_dif}
            
            # more details
            scl_keys = ["z_drv", "z_ship"]
            vec_keys = ["z_drv_od", "z_ship_j", "p"]
            for k in scl_keys:
                m_SO = solLP[n][k]
                m_FP = solFP[n][k]
                m_dif = np.abs((m_SO - m_FP) / m_SO)
                key_ = k + "_" + model_name
                metric.update({key_: m_dif})
            for k in vec_keys:
                m_SO = solLP[n][k]
                m_FP = solFP[n][k]
                m_dif = self.get_diff(m_SO, m_FP, k, model_name)
                metric.update({**m_dif})

            metrics.append(metric)
        return metrics

    def get_diff(self, a: np.ndarray, b: np.ndarray, name: str = "", model_name: str = ""):
        a = np.nan_to_num(a, nan=0)
        b = np.nan_to_num(b, nan=0)
        abs_dif = np.abs(a - b)
        deno = np.maximum(a,b) 
        rel_dif = abs_dif / np.abs(deno)
        nonzero_idx = np.where(deno > 0.)
        if nonzero_idx[0].shape[0] == 0:
            rdif_nz_max = 0.
            rdif_nz_mean = 0.
        else:
            rel_dif_nonzero = abs_dif[nonzero_idx] / deno[nonzero_idx]
            rdif_nz_max = rel_dif_nonzero.max()
            rdif_nz_mean = rel_dif_nonzero.mean()
        corr = np.corrcoef(a,b)[0,1]
        return {
            f"max_abs_dif_{name}_{model_name}": abs_dif.max(),
            f"mean_abs_dif_{name}_{model_name}": abs_dif.mean(),
            f"max_rel_dif_{name}_{model_name}": rel_dif.max(),
            f"mean_rel_dif_{name}_{model_name}": rel_dif.mean(),
            f"max_rel_dif_nonzero_{name}_{model_name}": rdif_nz_max,
            f"mean_rel_dif_nonzero_{name}_{model_name}": rdif_nz_mean,
            f"corr_{name}_{model_name}": corr,
        }