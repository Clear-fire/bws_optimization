#The code backbone is the dc3, which can refer https://github.com/locuslab/DC3
#You can use the bws_main to generate the novel DC3 backbone and then utilize the bws_second_main to train the second stage of the network 

try:
    import waitGPU
    waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=1000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass

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

from method import NNSolver
from method import dict_agg
from method import total_loss

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#wandb is used for convenient sharing
import wandb 
wandb.init(config={"learning_rate":0.01,},
           project="mywander method") 

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
    for k in range( hyper_arg['hyper_arr'].shape[0]): #k represents the number of experiments.
        hyper_arg['hyper_message'] = 'convex_ineq{}_eq{}_testdata'.format(args["simpleIneq"],args["simpleEq"])
        exp_stra = default_args.stratage_choose_args('dc31')
        my_train_net(data, args, save_dir, k+1, hyper_arg,exp_stra)


def my_train_net(data, args, save_dir, exp_time, hyper_arg,exp_stra):
    solver_step = 0.01  #args['lr'] 0.01 
    nepochs = args['epochs'] #nepochs = 300
    batch_size = args['batchSize'] #64
    num_steps = 5 #corr=10

    stratage_num = exp_stra["stratage_num"]
    is_corr_evaltrain = exp_stra['is_corr_evaltrain'] #whether check the loss variation of the correction during the training process.
    is_other_corr = exp_stra['is_other_corr'] #(deprecated)Is the newly added correction used during the testing process
    is_function_adjust = exp_stra['is_function_adjust'] #(deprecated)Does the objective function use adaptive gradients during the training process
    is_eval_withtrainingdata = exp_stra['is_eval_withtrainingdata'] #Is test data used for training
    is_store_net = exp_stra['is_store_net'] #Is the network saved
    net_store_step = exp_stra['net_store_step'] #AutoSave the network every few steps
    
    

    validX = data.testX #data.validX
    valid_dataset = TensorDataset(validX)
    if is_eval_withtrainingdata:
        train_dataset = TensorDataset(torch.cat([data.trainX, data.testX],dim=0))
    else:
        train_dataset = TensorDataset(data.trainX)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)

    solver_net = NNSolver(data, args)
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(solver_opt,milestones=[60,150,250],gamma=0.1) #FOR batch=128
    
    stats = {}
    str_arr = saved_str_arr(exp_stra,args)
    
    savedirpath = os.path.join(save_dir, "stratage{}_{}".format(stratage_num,exp_time))
    if not os.path.exists(savedirpath):
        os.makedirs(savedirpath)
    for i in range(nepochs):
        epoch_stats = {}
        # Get valid loss
        solver_net.eval()  
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_net(data, Xvalid, solver_net, args, 'valid_{}'.format(exp_stra['case']), epoch_stats,num_steps)

        # Get train loss
        solver_net.train()
        train_loss = 0
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            Yhat_train = solver_net(Xtrain)
            if num_steps==5:
                Ynew_train_arr = grad_steps_correct_5(data, Xtrain, Yhat_train, args)
            else:
                Ynew_train_arr = grad_steps_correct_10(data, Xtrain, Yhat_train, args)
            Ynew_train_arr = Ynew_train_arr.cuda(DEVICE)
            if is_corr_evaltrain: #Whether check the loss variation of the correction during the training process
                Ynew_train_arr2 = torch.cat([Yhat_train.view(1,Yhat_train.shape[0],-1), Ynew_train_arr],0)
                corr_eval(data, Xtrain,Ynew_train_arr2,'train_{}'.format(exp_stra['case']), epoch_stats)      
            if is_function_adjust:   
                train_loss = total_loss_adaptiveloss(data, Xtrain, Ynew_train_arr, args, exp_stra)
            else:
                train_loss = total_loss_muti(data, Xtrain, Ynew_train_arr, args)
            train_loss.sum().backward()
            solver_opt.step()
            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            #dict_agg(epoch_stats, 'train_time', train_time, op='sum')
        scheduler2.step()
        #print message
        print("learning rate is:",solver_opt.state_dict()['param_groups'][0]['lr'])
        trainingmessageprint(i,epoch_stats,exp_stra,num_steps)
        if (i%net_store_step==0 or i>290) and is_store_net:
            with open(os.path.join(savedirpath, 'solver_{}_net.dict'.format(i)), 'wb') as f:
                torch.save(solver_net.state_dict(), f)
    exp_inf_write(savedirpath,hyper_arg,exp_time,exp_stra)
    return solver_net, stats

#store the inequation and obkective message of the correction step   
@torch.no_grad()
def corr_eval(data, X, Ynew_arr, prefix, stats):
    make_prefix = lambda x: "{}_{}".format(prefix, x)
    stepnum = Ynew_arr.shape[0]
    ineq_bound_error=0.01
    for i in range(stepnum):
        dict_agg(stats, make_prefix('objfn_{}'.format(i)), data.obj_fn(Ynew_arr[i]).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_mean_{}'.format(i)), torch.mean(data.ineq_dist(X,Ynew_arr[i]), dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_max_{}'.format(i)), torch.max(data.ineq_dist(X,Ynew_arr[i]), dim=1)[0].detach().cpu().numpy())
        #calculate the inequation rate of bound
        ineq_dist=data.ineq_resid(X,Ynew_arr[i])
        ineq_size=ineq_dist.shape[0]*ineq_dist.shape[1]
        tmp = np.array(torch.tensor(ineq_dist[torch.abs(ineq_dist)<ineq_bound_error].shape[0]).view(1,1)/ineq_size)
        dict_agg(stats, make_prefix('ineq_bound_rate_{}'.format(i)), tmp)
        
#5 step correction
def grad_steps_correct_5(data, X, Y, args):
    take_grad_steps = args['useTrainCorr']
    num_steps = 5 #args['corrTrainSteps']
    Y_arr = torch.zeros((num_steps,Y.shape[0],Y.shape[1]))  #store Y
    if take_grad_steps:
        lr = args['corrLr']
        if args['probType']=='acopf57':
            momentum = args['corrMomentum']
        else:
            momentum = args['corrMomentum']*0.6
        partial_var = args['useCompl']
        partial_corr = True if args['corrMode'] == 'partial' else False
        if partial_corr and not partial_var:
            assert False, "Partial correction not available without completion."
        Y_new = Y
        old_Y_step = 0
        for i in range(num_steps):
            if partial_corr:
                if args['probType']=='acopf57':
                    Y_step = data.ineq_partial_grad(X, Y_new,lr)
                else:
                    Y_step = data.ineq_partial_grad(X, Y_new)
            else:
                ineq_step = data.ineq_grad(X, Y_new)
                eq_step = data.eq_grad(X, Y_new)
                Y_step = (1 - args['softWeightEqFrac']) * ineq_step + args['softWeightEqFrac'] * eq_step
            if args['probType']=='acopf57':
                #new_Y_step = data.momentun_fix(X,Y_new,Y_step, momentum, old_Y_step,lr)
                new_Y_step = lr * Y_step + momentum * old_Y_step
            else:
                new_Y_step = lr * Y_step + momentum * old_Y_step
            Y_new = Y_new - new_Y_step
            Y_arr[i,:,:]=Y_new
            old_Y_step = new_Y_step
        return Y_arr
    else:
        return Y
#10 step correction   
def grad_steps_correct_10(data, X, Y, args):
    take_grad_steps = args['useTrainCorr']
    num_steps = 10 #args['corrTrainSteps']
    Y_arr = torch.zeros((num_steps,Y.shape[0],Y.shape[1]))  #store Y
    if take_grad_steps:
        lr = args['corrLr']
        momentum = args['corrMomentum']
        partial_var = args['useCompl']
        partial_corr = True if args['corrMode'] == 'partial' else False
        if partial_corr and not partial_var:
            assert False, "Partial correction not available without completion."
        Y_new = Y
        old_Y_step = 0
        for i in range(num_steps):
            if partial_corr:
                Y_step = data.ineq_partial_grad(X, Y_new)
            else:
                ineq_step = data.ineq_grad(X, Y_new)
                eq_step = data.eq_grad(X, Y_new)
                Y_step = (1 - args['softWeightEqFrac']) * ineq_step + args['softWeightEqFrac'] * eq_step
            new_Y_step = lr * Y_step + momentum * old_Y_step
            Y_new = Y_new - new_Y_step
            Y_arr[i,:,:]=Y_new
            old_Y_step = new_Y_step
        return Y_arr
    else:
        return Y
#loss evaluation
def total_loss_muti(data, X, Y, args):
    obj_cost = data.obj_fn(Y[-1,:,:])     #objective minimum 
    ineq_dist = data.ineq_dist(X, Y[-1,:,:]) #inequation minimum
    #ineq_cost = torch.norm(ineq_dist, dim=1,p=1)  #norm 1 inequaiton
    ineq_cost = torch.norm(ineq_dist, dim=1) #norm 2 inequaiton
    #ineq_cost = torch.norm(ineq_dist, dim=1,p=2)**2 #l2 loss
    eq_cost = torch.norm(data.eq_resid(X, Y[-1,:,:]), dim=1) #equation loss
    return obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost + \
            args['softWeight'] * args['softWeightEqFrac'] * eq_cost
#(deprecated)zdaptive objective loss 
def total_loss_adaptiveloss(data, X, Y_arr, args,exp_stra):
    is_fn_obj = exp_stra['is_fn_obj']
    fn_grad = data.fn_partial_grad_nograd(X,Y_arr[-1])
    ineq_grad = data.ineq_partial_grad_nograd(X,Y_arr[-1])
    ineq_effective = loss_effective_gen(fn_grad,ineq_grad)
 
    obj_cost = data.obj_fn(Y_arr[-1])    
    ineq_dist = data.ineq_dist(X, Y_arr[-1]) 
    ineq_cost = torch.norm(ineq_dist, dim=1) 
    eq_cost = torch.norm(data.eq_resid(X, Y_arr[-1]), dim=1) 
    if is_fn_obj:
        return ineq_effective * obj_cost +  10*(1 - args['softWeightEqFrac']) * ineq_cost + \
            args['softWeightEqFrac'] * eq_cost
    else:
        return obj_cost + ineq_effective * (1 - args['softWeightEqFrac']) * ineq_cost + \
            args['softWeightEqFrac'] * eq_cost
#(deprecated)generate the correlation coefficient between two gradients.
@torch.no_grad()
def loss_effective_gen(based_grad,adaptive_grad): #the demension of adaptive_grad and based_grad are 833*100
    is_exp = True
    based_norm = torch.norm(based_grad,p=2,dim=1).view(-1,1) + 1e-6 #833*1
    base_stand = torch.div(based_grad,based_norm)
    adaptive_norm = torch.norm(adaptive_grad,p=2,dim=1).view(-1,1) + 1e-6  #833*1
    adaptive_stand = torch.div(adaptive_grad,adaptive_norm) #833*100
    effective_grad = torch.mul(base_stand,adaptive_stand).sum(axis=1)
    if is_exp:
        res_effective = torch.exp(effective_grad+1) #the value ranges from exp(-2) to 1
    return res_effective

def exp_inf_write(save_root,hyper_arg,exp_time,exp_stra):
    filename = 'arg.txt'
    with open(os.path.join(save_root,filename),'w') as f:
        for key in exp_stra:
            f.write('{} is {}\n'.format(key,exp_stra[key]))
        f.write('hyper_para is {}\n'.format(hyper_arg['hyper_arr'][exp_time-1]))
        f.write('hyper_message is {}\n'.format(hyper_arg['hyper_message']))

def saved_str_arr(exp_stra,args):
    is_corr_evaltrain = exp_stra['is_corr_evaltrain']
    case = exp_stra['case']
    is_other_corr =exp_stra['is_other_corr']
    str = []
    if is_corr_evaltrain:
        for i in range(args['corrTrainSteps']):
            str.append('train_{}_objfn_{}'.format(case,i))
            str.append('train_{}_ineq_mean_{}'.format(case,i))
    if is_other_corr:
        step = exp_stra['step'] 
        for i in range(step+1):
            str.append('valid_{}_objfn_{}'.format(case,i))
            str.append('valid_{}_ineq_mean_{}'.format(case,i))
    str.append('train_loss')
    str.append('valid_{}_eval'.format(case))
    str.append('valid_{}_ineq_max'.format(case))
    str.append('valid_{}_ineq_mean'.format(case))
    str.append('valid_{}_eq_max'.format(case))
    str.append('valid_{}_steps'.format(case))
    str.append('valid_{}_time'.format(case))
    return str

def trainingmessageprint(i,epoch_stats,exp_stra,step_num):
    #is_other_corrtrain = exp_stra['is_other_corrtrain']
    case = exp_stra['case']
    step = step_num+1 #####defaults['corrTestMaxSteps'] = 10
    is_other_corr =exp_stra['is_other_corr']
    is_corr_evaltrain = exp_stra['is_corr_evaltrain']
    if is_corr_evaltrain:
        for k in range(step):
            print(
                'Epoch {}:  train_fn_obj {:.4f},train_ineq_max {:.4f},ineq_rate {:.3%}'.format(
                    k, np.mean(epoch_stats['train_{}_objfn_{}'.format(case,k)]),np.mean(epoch_stats['train_{}_ineq_mean_{}'.format(case,k)]),
                    np.mean(epoch_stats['train_{}_ineq_bound_rate_{}'.format(case,k)])))
            """ print(
                'Epoch {}:  train_fn_obj {:.4f},train_ineq_max {:.4f}'.format(
                    k, np.mean(epoch_stats['train_{}_objfn_{}'.format(case,k)]),np.mean(epoch_stats['train_{}_ineq_mean_{}'.format(case,k)])
                    )) """
            wandb.log({
                "fn_obj_{}".format(k):np.mean(epoch_stats['train_{}_objfn_{}'.format(case,k)]),
                "ineq_max_{}".format(k):np.mean(epoch_stats['train_{}_ineq_mean_{}'.format(case,k)])})
        print(
            'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.6f}, eq max {:.6f}, time {:.4f}'.format(
            i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_{}_eval'.format(case)]),
            np.mean(epoch_stats['valid_{}_ineq_max'.format(case)]), np.mean(epoch_stats['valid_{}_eq_max'.format(case)]), 
            np.mean(epoch_stats['valid_{}_time'.format(case)])))
        
    else:
        if is_other_corr:
            step = exp_stra['step']
            print(
                'Epoch {}: train loss {:.4f}, fnbeg {:.4f}, ineqbeg {:.4f}, fnend {:.4f}, ineqend {:.4f}'.format(
                    i, np.mean(epoch_stats['train_loss']),
                    np.mean(epoch_stats['valid_{}_objfn_{}'.format(case,0)]),np.mean(epoch_stats['valid_{}_ineq_mean_{}'.format(case,0)]),
                    np.mean(epoch_stats['valid_{}_objfn_{}'.format(case,step)]),np.mean(epoch_stats['valid_{}_ineq_mean_{}'.format(case,step)])))
        else:
            print(
                'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.8f}, eq max {:.8f}, time {:.4f}'.format(
                    i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_{}_eval'.format(case)]),
                    np.mean(epoch_stats['valid_{}_ineq_max'.format(case)]), np.mean(epoch_stats['valid_{}_eq_max'.format(case)]), 
                    np.mean(epoch_stats['valid_{}_time'.format(case)])))
            wandb.log({"train loss":epoch_stats['train_loss'],"fn_obj":np.mean(epoch_stats['valid_{}_eval'.format(case)]),
                          "valid_ineq_max": np.mean(epoch_stats['valid_{}_ineq_max'.format(case)]), "valid_eq_max": np.mean(epoch_stats['valid_{}_eq_max'.format(case)]) })

def eval_net(data, X, solver_net, args, prefix, stats,num_steps):
    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Y = solver_net(X)
    base_end_time = time.time()
    
    if num_steps==5:
        Ynew_train_arr = grad_steps_correct_5(data, X, Y, args)
    else:
        Ynew_train_arr = grad_steps_correct_10(data, X, Y, args)
    Ynew_train_arr = Ynew_train_arr.cuda(DEVICE)

    Ycorr = Ynew_train_arr[-1,:,:]
    end_time = time.time()
    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Ycorr, args).detach().cpu().numpy()) #包括好几样的总损失，包括优化函数、不等式、等式约束
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy()) #优化的目标函数值
    dict_agg(stats, make_prefix('dist'), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy())
    return stats  
    
  
if __name__=='__main__':
    main()
   