import torch
import numpy as np
import math
import argparse
import pickle
import os

import time
import default_args
from setproctitle import setproctitle
from utils import my_hash, str_to_bool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device


def alt_diff(Pi, qi, Ai, bi, Gi, hi):  
    n, m, d = qi.shape[0], bi.shape[0], hi.shape[0]
    xk = torch.zeros(n).to(device).to(torch.float64)
    sk = torch.zeros(d).to(device).to(torch.float64)
    lamb = torch.zeros(m).to(device).to(torch.float64)
    nu = torch.zeros(d).to(device).to(torch.float64)
    
    
    dxk = torch.zeros((n, n)).to(device).to(torch.float64)
    dsk = torch.zeros((d, n)).to(device).to(torch.float64)
    dlamb = torch.zeros((m, n)).to(device).to(torch.float64)
    dnu = torch.zeros((d, n)).to(device).to(torch.float64)
    
    rho = 1
    thres = 1e-3
    R = - torch.linalg.inv(Pi + rho * Ai.T @ Ai + rho * Gi.T @ Gi)
    
    res = [1000, -100]
    
    ATb = rho * Ai.T @ bi.double()
    GTh = rho * Gi.T @ hi
    begin2 = time.time()

    while abs((res[-1]-res[-2])/res[-2]) > thres:
        iter_time_start = time.time()
        #print((Ai.T @ lamb).shape)
        xk = R @ (qi + Ai.T @ lamb + Gi.T @ nu - ATb + rho * Gi.T @ sk - GTh)
        
        dxk = R @ (torch.eye(n).to(device) + Ai.T @ dlamb + Gi.T @ dnu + rho * Gi.T @ dsk)
        
        sk = relu(- (1 / rho) * nu - (Gi @ xk - hi))
        dsk = (-1 / rho) * sgn(sk).to(device).reshape(d,1) @ torch.ones((1, n)).to(device) * (dnu + rho * Gi @ dxk)

        lamb = lamb + rho * (Ai @ xk - bi)
        dlamb = dlamb + rho * (Ai @ dxk)

        nu = nu + rho * (Gi @ xk + sk - hi)
        dnu = dnu + rho * (Gi @ dxk + dsk)

        res.append(0.5 * (xk.T @ Pi @ xk) + qi.T @ xk)      
    return (xk, dxk)

def cosDis(vec1, vec2):
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def decode(X_):
    a = []
    X = X_.numpy()
    for i in range(len(X)):
        a.append(X[i])
    return a

def relu(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = 0
    return ss

def sgn(s):
    ss = torch.zeros(len(s))
    for i in range(len(s)):
        if s[i]<=0:
            ss[i] = 0
        else:
            ss[i] = 1
    return ss

def proj(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = (ss[i] + math.sqrt(ss[i] ** 2 + 4 * 0.001)) / 2
    return ss

def main():
    #many terms in parser are not supported by our code
    parser = argparse.ArgumentParser(description='simple')
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

    setproctitle('DC3-{}'.format(args['probType'])) 
    #print(args)

    #initialize the hyperparameter ,
    #load data, and move the data to the GPU if needed
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
    for attr in dir(data):
        var = getattr(data, attr) 
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE)) 
            except AttributeError:
                pass
    data._device = DEVICE

    save_dir = os.path.join('results', str(data), 'method', my_hash(str(sorted(list(args.items())))),
                            str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hyper_arg = {}
    hyper_arg['hyper_arr'] = np.array([1])

    validX = data.testX
    datanum = validX.shape[0]
    dimN = data.ydim

    y_res=torch.zeros((datanum,dimN),dtype=torch.float64,device=device)


    time1 = time.time()
    for i in range(datanum):
        y_res[i], _ = alt_diff(Pi=data.Q, qi=data.p, Ai=data.A, bi=validX[i], Gi=data.G, hi=data.h)
    time2 = time.time()
    print("obj is {}".format(np.mean(data.obj_fn(y_res).detach().cpu().numpy())))
    print("max ineq is {}".format(np.mean(torch.max(data.ineq_dist(validX, y_res), dim=1)[0].detach().cpu().numpy())))
    print("max eq is {}".format(np.mean(torch.max(torch.abs(data.eq_resid(validX, y_res)), dim=1)[0].detach().cpu().numpy())))
    print("time is {}".format((time2-time1)/datanum))
    
    
if __name__=='__main__':
    main()
