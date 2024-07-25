try:
    import waitGPU
    waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=1000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)
import wandb

import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse

from utils import my_hash, str_to_bool
import default_args

from method import NNSolver
from method import dict_agg
from method import total_loss 
from qpsolvers import available_solvers, print_matrix_vector, solve_qp

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#DEVICE = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description='DC3')
    parser.add_argument('--probType', type=str, default='simple',
                        choices=['simple', 'nonconvex', 'acopf57'], help='problem type')
    parser.add_argument('--simpleVar', type=int,
                        help='number of decision vars for simple problem')
    parser.add_argument('--simpleIneq', type=int,
                        help='number of inequality constraints for simple problem')
    parser.add_argument('--simpleEq', type=int,
                        help='number of equality constraints for simple problem')
    parser.add_argument('--simpleEx', type=int,
                        help='total number of datapoints for simple problem')
    parser.add_argument('--nonconvexVar', type=int,
                        help='number of decision vars for nonconvex problem')
    parser.add_argument('--nonconvexIneq', type=int,
                        help='number of inequality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEq', type=int,
                        help='number of equality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEx', type=int,
                        help='total number of datapoints for nonconvex problem')
    parser.add_argument('--epochs', type=int,
                        help='number of neural network epochs')
    parser.add_argument('--batchSize', type=int,
                        help='training batch size')
    parser.add_argument('--lr', type=float,
                        help='neural network learning rate')
    parser.add_argument('--hiddenSize', type=int,
                        help='hidden layer size for neural network')
    parser.add_argument('--softWeight', type=float,
                        help='total weight given to constraint violations in loss')
    parser.add_argument('--softWeightEqFrac', type=float,
                        help='fraction of weight given to equality constraints (vs. inequality constraints) in loss')
    parser.add_argument('--useCompl', type=str_to_bool,
                        help='whether to use completion')
    parser.add_argument('--useTrainCorr', type=str_to_bool,
                        help='whether to use correction during training')
    parser.add_argument('--useTestCorr', type=str_to_bool,
                        help='whether to use correction during testing')
    parser.add_argument('--corrMode', choices=['partial', 'full'],
                        help='employ DC3 correction (partial) or naive correction (full)')
    parser.add_argument('--corrTrainSteps', type=int,
                        help='number of correction steps during training')
    parser.add_argument('--corrTestMaxSteps', type=int,
                        help='max number of correction steps during testing')
    parser.add_argument('--corrEps', type=float,
                        help='correction procedure tolerance')
    parser.add_argument('--corrLr', type=float,
                        help='learning rate for correction procedure')
    parser.add_argument('--corrMomentum', type=float,
                        help='momentum for correction procedure')
    parser.add_argument('--saveAllStats', type=str_to_bool,
                        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int,
                        help='how frequently (in terms of number of epochs) to save stats to file')

    args = parser.parse_args()
    args = vars(args)  # change to dictionary
    defaults = default_args.method_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    # print(args)
    ######初始化所有值
    setproctitle('DC3-{}'.format(args['probType']))  # 修改进程名称
    #####Load data, and to GPU if needed，根据probtype选择数据种类
    prob_type = args['probType']
    if prob_type == 'simple':
        filepath = os.path.join('datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif prob_type == 'nonconvex':
        filepath = os.path.join('datasets', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    elif prob_type == 'acopf57':
        filepath = os.path.join('datasets', 'acopf', 'acopf57_dataset')
    else:
        raise NotImplementedError

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
   
    data_test=data.testX
    if args['probType']=="acopf57":
        y_groundtruth=0
    else:
        y_groundtruth=torch.tensor(gen_ans(data))
        #y_groundtruth= torch.tensor(y_groundtruth).cuda(DEVICE)
    Ynew_arr=y_groundtruth
    print("objfn is:{}".format(np.mean(data.obj_fn(Ynew_arr).detach().cpu().numpy())))
    print("ineq_max is:{}".format(np.mean(torch.max(data.ineq_dist(data_test,Ynew_arr), dim=1)[0].detach().cpu().numpy())))
    print("eq_max is:{}".format(np.mean(torch.max(torch.abs(data.eq_resid(data_test, Ynew_arr)), dim=1)[0].detach().cpu().numpy())))
        
      
    

@torch.no_grad()     
def gen_ans(data):
    P = np.array(data.Q.to("cpu"))
    #P = torch.randn(100,100)
    q = np.array(data.p.to("cpu"))
    A = np.array(data.A.to("cpu"))
    
    G = np.array(data.G.to("cpu"))
    h = np.array(data.h.to("cpu"))
    solver ="osqp"
    mydata =data.testX

    datanum = mydata.shape[0]
    x = np.zeros((datanum,100))
    testX = mydata
    time1=time.time()
    for i in range(datanum):
        b = np.array(testX[i,:].to("cpu"))
        x[i,:] = solve_qp(P, q, G, h, A, b, solver=solver)
    time2=time.time()
    print("time is:{}".format((time2-time1)/833))   
    return x
           
if __name__=='__main__':
    main()
   