"""network.py
Perform network loading on a task-chain network
"""

import numpy as np
import pandas as pd
from dataset import *

class MNL(object):
    def __init__(self,
                 param_: Param,
                 dataset: Data
                 ) -> None:

        self.eps = 1e-8

        # parameter
        self.W = param_.nOD
        self.J = param_.nRS
        self.T = param_.T
        self.K = param_.K
        self.theta = param_.theta
        self.phi = param_.phi # * (param_.K+1)
        
        # additional params
        self.td = self.J
        self.to = self.J + 1

        # read data
        self.OD = dataset.OD
        self.RS = dataset.RS
        self.c_drv = dataset.c_drv # W x J+2 x J+2 or W x J+1 x J+1
        self.c_ship = dataset.c_ship # J x T+1
        self.drv_flow = dataset.drv_flow # W x T
        self.ship_flow = dataset.ship_flow # J x 1
        self.co = dataset.co # D x Od x J+1

        # additional data
        self.dests = np.unique(self.OD[:,1])
        self.D = self.dests.shape[0]
        
    def compute_y(self, p: np.ndarray) -> np.ndarray:
        # compute time-window choice costs
        y = np.zeros((self.T+1, self.J), dtype=np.float64)
        S = np.zeros((self.J,), dtype=np.float64)
        for j in range(self.J):
            yj, Sj = self.compute_yj(self.c_ship[j], p[:,j], self.ship_flow[j])
            y[:,j] = yj
            S[j] = Sj
        return y, S

    def compute_yj(self, c: np.ndarray, p: np.ndarray, N: float) -> np.ndarray:
        # preparation
        px = np.concatenate([p, [0.]]) # for opt-out, then T+1 length
        M = np.exp(-self.theta * (c + px)) # T+1 x 1
        # probability and flow
        q = M / np.sum(M, keepdims=True)
        yj = N * q
        Sj = N * (-np.log(M.sum())/self.theta)
        return yj, Sj

# od-specific assignment with d-dataset
class MNL_od(MNL):

    def __init__(self,
                 param_: Param,
                 dataset: Data,
                 ) -> None:
        
        super().__init__(param_, dataset)
        self.d_idxs = {d: np.where(self.OD[:,1] == self.dests[d])[0] for d in range(self.D)}
        self.keys_ = [] # list of tuple (t, w, o, d)
        for d in range(self.D):
            for o, w in enumerate(self.d_idxs[d]):
                for t in range(self.T):
                    if (t, w, o, d) not in self.keys_:
                        self.keys_.append((t, w, o, d))
            
    def compute_x(self, p: np.ndarray) -> np.ndarray:
        x = {}
        x_task = np.zeros((self.T, self.W, self.J), dtype=np.float64)
        S = np.zeros((self.T, self.W), dtype=np.float64) # surplus
        for (t, w, o, d) in self.keys_:
            xo_tw, x_tw, xtask_tw, Stw = self.compute_x_od(self.c_drv[d], self.co[d][o], p[t], self.drv_flow[d][t,o])
            x[(t,w)] = [xo_tw, x_tw]
            x_task[t,w] = xtask_tw
            S[t,w] = Stw
        return x, x_task, S

    def compute_x_od(self, c: np.ndarray, co: np.ndarray, p: np.ndarray, N: float) -> np.ndarray:           
        # preparation
        px = np.concatenate([p, [0.]]) # for destination, then J+1 length
        M = np.exp(-self.phi * (c - px[np.newaxis,:])) # J+1 x J+1
        M[self.td,:self.td] = 0
        Mo = np.exp(-self.phi * (co - px)) # J+1 x 1

        ## Step 1: compute value function V
        V = np.zeros((self.K+1, self.J+1), dtype=np.float64) # from k=1 to k=K+1
        V[:,self.td] = 1 # (k,d) for all k = 1 to K+1
        k = self.K - 1
        while True:
            if k == -1:
                Vo = Mo.dot(V[0,:])
                break
            # for k > 1
            for j in range(self.J):
                V[k,j] = M[j,:].dot(V[k+1,:])
            # update k
            k -= 1
    
        ## Step 2: compute link flows x
        z = np.zeros((self.K+1, self.J+1), dtype=np.float64) # state flows
        x = np.zeros((self.K+1, self.J+1, self.J+1), dtype=np.float64) # edge flows
        # for departing edges
        xo = N * (Mo * V[0,:]) / Vo # K+1 x 1
        # for first tasks
        z[0,:] = xo
        for j in range(self.J+1):
            x[0,j,:] = z[0,j] * (M[j,:] * V[1,:]) / V[0,j]
        # then forward computation
        k = 1
        while True:
            z[k,self.td] = x[k-1,:,self.td].sum()
            if k == self.K:
                assert np.round(z[k,self.td]) == N, f"Absorbed flow is not equal to total N!! Err. = {N} - {z[k,self.td]}"
                break
            # for k < K+1
            x[k,self.td,self.td] = z[k,self.td] # all flow to d with prob. of one
            for j in range(self.J):
                z[k,j] = x[k-1,:,j].sum()
                x[k,j,:] = z[k,j] * (M[j,:] * V[k+1,:]) / V[k,j]
            # update k
            k += 1
        
        # number of drivers performing each task
        x_task = z.sum(axis=0) # more simple than summing up x

        # surplus
        S = N * -np.log(Vo)/self.phi
        return xo, x, x_task[:self.J], S
    
    def compute_S_drv(self, p: np.ndarray) -> np.ndarray:
        S = np.zeros((self.T, self.W), dtype=np.float64)
        for d in range(self.D):
            for o, w in enumerate(self.d_idxs[d]):
                for t in range(self.T):
                    S[t,w] = self.compute_Vod(self.c_drv[d], self.co[d][o], p[t], self.drv_flow[d][t,o])
        return S

    def compute_Vod(self, c: np.ndarray, co: np.ndarray, p: np.ndarray, N: float) -> float:
        # preparation
        px = np.concatenate([p, [0.]]) # for destination, then J+1 length
        M = np.exp(-self.phi * (c - px[np.newaxis,:])) # J+1 x J+1
        M[self.td,:self.td] = 0
        Mo = np.exp(-self.phi * (co - px[np.newaxis,:])) # J+1 x 1

        ## Step 1: compute value function V
        # computation of exp(s)
        V = np.zeros((self.K+1, self.J+1), dtype=np.float64) # from k=1 to k=K+1
        V[:,self.td] = 1 # (k,d) for all k = 1 to K+1
        k = self.K - 1
        while True:
            if k == -1:
                Vo = Mo.dot(V[0,:])
                break
            # for k > 1
            for j in range(self.J):
                V[k,j] = M[j,:].dot(V[k+1,:])
            # update k
            k -= 1
        # surplus
        S = N * -np.log(Vo)/self.phi
        return S

class MNL_d(MNL):

    def __init__(self,
                 param_: Param,
                 dataset: Data,
                 ) -> None:
        
        super().__init__(param_, dataset)
        
    def compute_x(self, p: np.ndarray) -> np.ndarray:
        x = {}
        x_task = np.zeros((self.T, self.D, self.J))
        S = np.zeros((self.T, self.D))
        for d in range(self.D):
            for t in range(self.T):
                xo_td, x_td, xtask_td, Std = self.compute_xd(
                    self.c_drv[d], self.co[d], self.drv_flow[d][t], p[t])
                x[(t,d)] = [xo_td, x_td]
                x_task[t,d] = xtask_td
                S[t,d] = Std
        return x, x_task, S

    def compute_xd(self, c: np.ndarray, co: np.ndarray, Ntd: np.ndarray, p: np.ndarray) -> np.ndarray:           
        # preparation
        O = co.shape[0] # can be different across d, so not constant (self) parameter
        px = np.concatenate([p, [0.]]) # for destination, then J+1 length
        M = np.exp(-self.phi * (c - px[np.newaxis,:])) # J+1 x J+1
        M[self.td,:self.td] = 0
        Mo = np.exp(-self.phi * (co - px[np.newaxis,:])) # O x J+1

        ## Step 1: compute value function V
        # computation of exp(s)
        V = np.zeros((self.K+1, self.J+1), dtype=np.float64) # from k=1 to k=K+1
        V[:,self.td] = 1 # (k,d) for all k = 1 to K+1
        k = self.K - 1
        while True:
            if k == -1:
                Vo = Mo.dot(V[0,:])
                break
            # for k > 1
            for j in range(self.J):
                V[k,j] = M[j,:].dot(V[k+1,:])
            # update k
            k -= 1
    
        ## Step 2: compute link flows x
        z = np.zeros((self.K+1, self.J+1), dtype=np.float64) # state flows
        x = np.zeros((self.K+1, self.J+1, self.J+1), dtype=np.float64) # edge flows
        # for departing edges
        xo = np.zeros((O, self.J+1), dtype=np.float64) # departing edge flows
        for o in range(O):
            xo[o,:] = Ntd[o] * (Mo[o,:] * V[0,:]) / Vo[o]
        # for first tasks
        z[0,:] = xo.sum(axis=0)
        for j in range(self.J+1):
            x[0,j,:] = z[0,j] * (M[j,:] * V[1,:]) / V[0,j]
        # then forward computation
        k = 1
        while True:
            z[k,self.td] = x[k-1,:,self.td].sum()
            if k == self.K:
                assert np.round(z[k,self.td]) == Ntd.sum(), f"Absorbed flow is not equal to total N!! Err. = {Ntd.sum()} - {z[k,self.td]}"
                break
            # for k < K+1
            x[k,self.td,self.td] = z[k,self.td] # all flow to d with prob. of one
            for j in range(self.J):
                z[k,j] = x[k-1,:,j].sum()
                x[k,j,:] = z[k,j] * (M[j,:] * V[k+1,:]) / V[k,j]
            # update k
            k += 1
        
        # number of drivers performing each task
        x_task = z.sum(axis=0) # more simple than summing up x

        # surplus
        logV = -np.log(Vo)/self.phi
        S = np.dot(Ntd, logV)
        return xo, x, x_task[:self.J], S
    
    def compute_S_drv(self, p: np.ndarray) -> np.ndarray:
        S = np.zeros((self.T, self.D), dtype=np.float64)
        for d in range(self.D):
            for t in range(self.T):
                Vd = self.compute_Vd(self.c_drv[d], self.co[d], p[t])
                logV = -np.log(Vd)/self.phi
                S[t,d] = np.dot(self.drv_flow[d][t], logV)
        return S

    def compute_Vd(self, c: np.ndarray, co: np.ndarray, p: np.ndarray) -> np.ndarray:
        # preparation
        px = np.concatenate([p, [0.]]) # for destination, then J+1 length
        M = np.exp(-self.phi * (c - px[np.newaxis,:])) # J+1 x J+1
        M[self.td,:self.td] = 0
        Mo = np.exp(-self.phi * (co - px[np.newaxis,:])) # O x J+1

        ## Step 1: compute value function V
        # computation of exp(s)
        V = np.zeros((self.K+1, self.J+1), dtype=np.float64) # from k=1 to k=K+1
        V[:,self.td] = 1 # (k,d) for all k = 1 to K+1
        k = self.K - 1
        while True:
            if k == -1:
                Vo = Mo.dot(V[0,:])
                break
            # for k > 1
            for j in range(self.J):
                V[k,j] = M[j,:].dot(V[k+1,:])
            # update k
            k -= 1
        return Vo
