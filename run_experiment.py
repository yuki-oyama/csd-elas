import os
from dataset import *
from network import MNL_od
from model import CSD
import numpy as np
import pandas as pd
import json
import time
from dataclasses import asdict
from tqdm import tqdm
from utils import Timer
import argparse

#### argparse ####
parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
  return v.lower() in ('true', '1')

def float_or_none(value):
    try:
        return float(value)
    except:
        return None

# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--seed', type=int, default=123, help='random seed')
model_arg.add_argument('--root', type=str, default=None, help='root directory to save results')
model_arg.add_argument('--out_dir', type=str, default='test', help='output directory to be created')
model_arg.add_argument('--nExp', type=int, default=10, help='number of experiments with the same parameter')
model_arg.add_argument('--nOD', type=int, default=10, help='number of OD pairs')
model_arg.add_argument('--nRS', type=int, default=10, help='number of RS pairs')
model_arg.add_argument('--nDrv', type=int, default=10000, help='number of drivers')
model_arg.add_argument('--nShp', type=int, default=10000, help='number of tasks')
model_arg.add_argument('--T', type=int, default=4, help='number of time windows')
model_arg.add_argument('--K', type=int, default=2, help='max number of tasks a driver can perform')
model_arg.add_argument('--theta', type=float, default=1.0, help='theta')
model_arg.add_argument('--phi', type=float, default=1.0, help='phi')
model_arg.add_argument('--paramName', nargs='+', type=str, default=['nOD'], help='name of parameter to change in experiment')
model_arg.add_argument('--paramVals', nargs='+', type=float, default=[10], help='parameter values to change in experiment')
model_arg.add_argument('--od_weight', type=str2bool, default=False, help='if sample OD pairs with weights or not')
model_arg.add_argument('--rs_weight', type=str2bool, default=False, help='if sample RS pairs with weights or not')
model_arg.add_argument('--t_distrb', type=str, default="Uniform", help='driver temporal distribution')
model_arg.add_argument('--gamma', type=float, default=1.0, help='coefficient of c-bar, professional delivery cost')
model_arg.add_argument('--speed', type=float, default=1.0, help='normal vehicle speed in miles per hour')
model_arg.add_argument('--vot', type=float, default=4.0, help='value of time in usd per hour')
model_arg.add_argument('--base_c', type=float, default=0.0, help='base cost')
model_arg.add_argument("--compare_naive", type=str2bool, default=False, help='whether to run N0 or not')
agd_arg = add_argument_group('AGD')
agd_arg.add_argument('--fp_tol', type=float, default=1e-1, help='tolerance for master problem: max gradient gap')
agd_arg.add_argument('--g0', type=float, default=1e-3, help='initial step size of AGD')
agd_arg.add_argument('--gmin', type=float, default=1e-6, help='minimum step size of AGD')
agd_arg.add_argument('--maxIter', type=float, default=1000, help='maximum number of iterations of AGD')
agd_arg.add_argument('--backIter', type=float, default=1000, help='number of iterations to start Backtracking')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed


if __name__ == '__main__':
    config, _ = get_config()
    timer = Timer()
    np.random.seed(config.seed)

    # output directory
    if config.root is not None:
        out_dir = os.path.join(config.root, "results", config.out_dir)
    else:
        out_dir = os.path.join("results", config.out_dir)
    
    try:
        os.makedirs(out_dir, exist_ok = False)
    except:
        out_dir += '_' + time.strftime("%Y%m%dT%H%M")
        os.makedirs(out_dir, exist_ok = False)

    # save config
    with open(os.path.join(out_dir, "config.json"), mode="w") as f:
            json.dump(config.__dict__, f, indent=4)

    # parameter setting
    nOD = config.nOD #100
    nRS = config.nRS #100
    nDrv = config.nDrv #50000
    nShp = config.nShp #50000
    theta = config.theta #1
    phi = config.phi #1
    T = config.T
    K = config.K
    base = {"nOD": nOD, "nRS": nRS, "nDrv": nDrv, "nShp": nShp, 
                "T": T, "K": K, "theta": theta, "phi": phi}

    for val_ in config.paramVals:
        param_ = setparam_(nOD, nRS, nDrv, nShp, T, K, theta, phi)
        name = config.paramName[0]
        if type(base[name]) == int:
            param_.__dict__[name] = int(val_)
        elif type(base[name]) == float:
            param_.__dict__[name] = float(val_)
        if name == 'nShp':
            param_.__dict__['nDrv'] = int(val_)
        if name == 'theta':
            param_.__dict__['phi'] = float(val_)
        print(param_)

        # set data
        link_path = 'data/Winnipeg/link.csv'
        od_path = 'data/Winnipeg/od.csv'
        cSP_load = np.load('data/Winnipeg/cSP.npy')
        dataset = Dataset(link_path, od_path, cSP_load, od_weight=config.od_weight, rs_weight=config.rs_weight,
                          t_distrb=config.t_distrb, gamma=config.gamma, speed=config.speed, vot=config.vot,
                          base_c=config.base_c)
        expData = dataset.generate_data(param_, N = config.nExp)

        # define model
        csd = CSD(nw_model=MNL_od, solver='gurobi', nThrd=32, msg=False, solve_dual=True, 
                  accuracy=config.fp_tol, init_step_size=config.g0, min_step_size=config.gmin,
                  maxIter=config.maxIter, backIter=config.backIter)
        csd.load_data(param_, expData)

        # solve LP
        recordsLP, solLP = csd.solve_SOP_lp()
        dfResLP = pd.DataFrame(recordsLP)

        # solve LP by N0
        if config.compare_naive:
            recordsN0, solN0 = csd.solve_SOP_lp_N0()
            dfResN0 = pd.DataFrame(recordsN0)
            
            # evaluate metrics - N0
            metrics_N0 = csd.compare_results(recordsLP, solLP, recordsN0, solN0, "N0")
            dfMetrics_N0 = pd.DataFrame(metrics_N0)

        # solve FP
        recordsFP, solFP = csd.solve_fluid_particle()
        dfResFP = pd.DataFrame(recordsFP)      
        
        # evaluate metrics - FP
        metrics_fp = csd.compare_results(recordsLP, solLP, recordsFP, solFP, "FP")
        dfMetrics_fp = pd.DataFrame(metrics_fp)
        
        # save results
        file_path = os.path.join(out_dir, f"res_{str(val_)}.csv")
        if config.compare_naive:
            dfRes = pd.concat([dfResLP, dfResN0, dfResFP, dfMetrics_N0, dfMetrics_fp], axis=1)
        else:
            dfRes = pd.concat([dfResLP, dfResFP, dfMetrics_fp], axis=1)
        for name in config.paramName:
            dfRes[name] = val_
        print(dfRes)
        dfRes.to_csv(file_path, index=False)

        # save param_ as json
        with open(os.path.join(out_dir, f"param_{str(val_)}.json"), mode="w") as f:
            json.dump(asdict(param_), f)
