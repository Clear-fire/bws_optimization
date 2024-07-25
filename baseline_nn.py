try:
    import waitGPU
    waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass

from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

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

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
import wandb
wandb.init(config={"learning_rate":0.01,},
           project="qp-layer-NN") 
WANDB_START_METHOD="thread"

def main():
    parser = argparse.ArgumentParser(description='baseline_nn')
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
    parser.add_argument('--useTestCorr', type=str_to_bool,
        help='whether to use correction during testing')
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
    args = vars(args) # change to dictionary
    defaults = default_args.baseline_nn_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)

    setproctitle('baselineNN-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        torch.set_default_dtype(torch.float64)
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

    save_dir = os.path.join('results', str(data), 'baselineNN',
        my_hash(str(sorted(list(args.items())))), str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)

    # Run pure neural network baseline
    solver_net, stats = train_net(data, args, save_dir)



def train_net(data, args, save_dir):
    solver_step = 0.01 #
    nepochs = 400
    batch_size = 64 #64
    is_store_net = True
    is_test_reload_net = True

    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    solver_net = NNSolver_two(data, args)
    #solver_net = NNSolver(data, args)
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(solver_opt,milestones=[160,220,350],gamma=0.1)

    stats = {}
    A_inv, partial_vars = matrix_gen(data, args)
    A_inv.detach()
    partial_vars.detach()
    savedirpath = os.path.join(save_dir, "nn_net{}".format(1))
    #载入测试
    #if 
    for i in range(nepochs):
        epoch_stats = {}

        """ # Get valid loss
        solver_net.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_net(data, Xvalid, solver_net, args, 'valid', epoch_stats) """

        # Get test loss
        solver_net.eval()
        for Xtest in test_loader:
            Xtest = Xtest[0].to(DEVICE)
            eval_net(data, Xtest, solver_net, args, 'valid', epoch_stats) 
 
        # Get train loss
        solver_net.train()
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            Yhat_train = solver_net(Xtrain)
            if i<=300:
                train_loss = softloss_adp(data, Xtrain, Yhat_train, args,i)
                #train_loss = softloss(data, Xtrain, Yhat_train, args)
            else:
                Y_new = completion_gen(data, Xtrain, Yhat_train, partial_vars, A_inv)
                train_loss = softloss_adp(data, Xtrain, Y_new, args,i)
            train_loss.sum().backward()
            solver_opt.step()
            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')

        scheduler2.step()
        print("learning rate is:",solver_opt.state_dict()['param_groups'][0]['lr'])
        # Print results
        print(
            'Epoch {}: train loss {:.4f}, eval {:.4f}, dist {:.4f}, ineq max {:.8f},  eq max {:.6f}, time {:.4f}'.format(
                i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                np.mean(epoch_stats['valid_dist']), np.mean(epoch_stats['valid_ineq_max']),
                np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_time'])))
        wandb.log({"eval":np.mean(epoch_stats['valid_eval']),"eq_max":np.mean(epoch_stats['valid_eq_max']),
                                  "ineq_max": np.mean(epoch_stats['valid_ineq_max'])})

        if args['saveAllStats']:
            if i == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        else:
            stats = epoch_stats

        if (i % args['resultsSaveFreq'] == 0):
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
                torch.save(solver_net.state_dict(), f)

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    

    return solver_net, stats

# Modifies stats in place
def dict_agg(stats, key, value, op='concat'):
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value

# Modifies stats in place
def eval_net(data, X, solver_net, args, prefix, stats):

    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)
    start_time = time.time()
    Y = solver_net(X)
    raw_end_time = time.time()
    Ycorr=Y
    #Ycorr, steps = grad_steps_all(data, X, Y, args)

    dict_agg(stats, make_prefix('time'), time.time() - start_time, op='sum')
    dict_agg(stats, make_prefix('loss'), softloss(data, X, Y, args).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy())
    dict_agg(stats, make_prefix('dist'), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
  

    return stats

def softloss(data, X, Y, args):
    obj_cost = data.obj_fn(Y)
    ineq_cost = torch.norm(data.ineq_dist(X, Y), dim=1)
    eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
    
    return obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost + \
           args['softWeight'] * args['softWeightEqFrac'] * eq_cost *0.2 

def softloss_adp(data, X, Y, args, epoch):
    obj_cost = data.obj_fn(Y)
    ineq_cost = torch.norm(data.ineq_dist(X, Y), dim=1)
    eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
    arr=[30,60,120,200]
    if epoch<arr[0]:
        eff=10
    elif epoch<arr[1]:
        eff=5
    elif epoch<arr[2]:
        eff=2
    elif epoch<arr[3]:
        eff=2
    else:
        eff=1
    
    return obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost *eff+ \
           args['softWeight'] * args['softWeightEqFrac'] * eq_cost *0.2 


def softloss1(data, X, Y, args):
    obj_cost = data.obj_fn(Y)
    ineq_cost = torch.norm(data.ineq_dist(X, Y), dim=1)
    eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
    return obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost + \
           args['softWeight'] * args['softWeightEqFrac'] * eq_cost *20

# Used only at test time, so let PyTorch avoid building the computational graph
def grad_steps_all(data, X, Y, args):
    take_grad_steps = args['useTestCorr']
    if take_grad_steps:
        lr = args['corrLr']
        eps_converge = args['corrEps']
        max_steps = args['corrTestMaxSteps']
        momentum = args['corrMomentum']

        Y_new = Y
        i = 0
        old_step = 0
        with torch.no_grad():
            while (i == 0 or torch.max(torch.abs(data.eq_resid(X, Y_new))) > eps_converge or
                           torch.max(data.ineq_dist(X, Y_new)) > eps_converge) and i < max_steps:
                with torch.no_grad():
                    ineq_step = data.ineq_grad(X, Y_new)
                    eq_step = data.eq_grad(X, Y_new)
                    Y_step = (1 - args['softWeightEqFrac']) * ineq_step + args['softWeightEqFrac'] * eq_step
                    new_step = lr * Y_step + momentum * old_step
                    Y_new = Y_new - new_step
                    old_step = new_step
                    i += 1
        return Y_new, i
    else:
        return Y, 0

def get_equation_inverse(data):
    return data._A_other_inv

def softloss_core(data, X, Y, args):
    obj_cost = data.obj_fn(Y)
    ineq_cost = torch.norm(data.ineq_dist(X, Y), dim=1)
    eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
    myeq_cost = eq_adaption.apply
    return obj_cost*0 + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost + myeq_cost(data, X, Y) + args['softWeight'] * args['softWeightEqFrac'] * eq_cost 

class eq_adaption(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, X, Y):
        ctx.save_for_backward(Y)
        eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
        ctx.data = data
        ctx.X=X
        return eq_cost

    @staticmethod
    def backward(ctx,grad_output):
        Y, = ctx.saved_tensors
        grad_output=torch.zeros_like(Y,device=DEVICE)
        data= ctx.data
        A=data.A
        G=ctx.X - Y[:,data.partial_vars]@A[:,data.partial_vars].T
        inv=data._A_other_inv
        grad_output[:,data.other_vars]=G @ inv
        #print(grad_output[5,data.other_vars[1]])
        return None,None,grad_output*0.1

def matrix_gen(data, args):
    A = data.A
    neq = data.neq
    ydim = data.ydim
    det = 0
    i = 0
    inv_matrix = 0
    while abs(det) < 0.0001 and i < 100: # abs(det) < 0.0001 and i < 100:
            partial_vars = np.random.choice(ydim,  neq, replace=False)
            det = torch.det(A[:, partial_vars]) #主要为了防止存在相关变量
            i += 1
    if i == 100:
        raise Exception
    else: 
        A_inv = torch.inverse(A[:, partial_vars])
        return A_inv, partial_vars

def completion_gen(data, X, Y, partial_vars, A_inv):
    eq_res = data.eq_resid(X, Y)
    eq_res.detach()
    Y[:, partial_vars] = Y[:, partial_vars] + eq_res * A_inv
    return  Y
    
######### Models

class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize']-150, self._args['hiddenSize']-150]
        """ layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.0)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])]) """
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        """ layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.LeakyReLU(0.1), nn.Dropout(p=0.0)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])]) """
        
        layers += [nn.Linear(layer_sizes[-1], data.ydim)]
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        prob_type = self._args['probType']
        if prob_type == 'simple':
            return self.net(x)
        elif prob_type == 'nonconvex':
            return self.net(x)
        elif 'acopf' in prob_type:
            out = self.net(x)
            data = self._data
            out2 = nn.Sigmoid()(out[:, :-data.nbus])
            pg = out2[:, :data.ng] * data.pmax + (1-out2[:, :data.ng]) * data.pmin
            qg = out2[:, data.ng:2*data.ng] * data.qmax + (1-out2[:, data.ng:2*data.ng]) * data.qmin
            vm = out2[:, 2*data.ng:] * data.vmax + (1- out2[:, 2*data.ng:]) * data.vmin
            return torch.cat([pg, qg, vm, out[:, -data.nbus:]], dim=1)
        else:
            raise NotImplementedError

class NNSolver_two(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize']]
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU()] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        """ layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])  """
        layers += [nn.Linear(layer_sizes[-1], data.ydim)]
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        prob_type = self._args['probType']
        if prob_type == 'simple':
            return self.net(x)
        elif prob_type == 'nonconvex':
            return self.net(x)
        elif 'acopf' in prob_type:
            out = self.net(x)
            data = self._data
            out2 = nn.Sigmoid()(out[:, :-data.nbus])
            pg = out2[:, :data.ng] * data.pmax + (1-out2[:, :data.ng]) * data.pmin
            qg = out2[:, data.ng:2*data.ng] * data.qmax + (1-out2[:, data.ng:2*data.ng]) * data.qmin
            vm = out2[:, 2*data.ng:] * data.vmax + (1- out2[:, 2*data.ng:]) * data.vmin
            return torch.cat([pg, qg, vm, out[:, -data.nbus:]], dim=1)
        else:
            raise NotImplementedError

if __name__=='__main__':
    main()