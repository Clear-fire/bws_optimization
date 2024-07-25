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
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#DEVICE = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")

wandb.init(config={"learning_rate":0.001,},
           project="mywander method") 


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
    setproctitle('DC3-{}'.format(args['probType']))  
    
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
    
    #filename=["1691506306-6322715","1691506433-6985471","1691506439-8689823","1691506460-8277786","1691506603-0387285"] #10*50
    #filename=["1691504849-5907636","1691504870-9026377","1691504908-6266408","1691504920-27085","1691504931-9349513"] #30*50
    filename=["1691507937-4205","1691508017-794747","1691508446-5036035","1691508530-1980238","1691508622-6948495"] # 50*50
    #filename=["1691509514-1302776","1691509714-9298234","1691509733-6383643","1691509795-4247196","1691509807-9751022"] #70*50
    #filename=["1691510692-8263044","1691510763-8775425","1691510777-8635094","1691510845-255128","1691510861-2799058"] #90*ineq50
    #filename=["1691565359-312683","1691565457-0746434","1691565492-2076724","1691565499-1463935","1691565519-440277"] #50*10
    #filename=["1691566763-00561","1691566778-2911031","1691566794-867234","1691566811-8484774","1691567023-1378179"] #50*30 
    #filename=["1691568035-2453105","1691568083-49865","1691568194-2898006","1691568211-002718","1691568227-8791013"] #50*70
    #filename=["1691569226-3209813","1691569255-556839","1691569280-7950559","1691569343-0234818","1691569388-496077"] #50*90
    #filename=["1691655219-842858","1691655274-077214","1691655290-4794054","1691655300-5201285","1691655310-254908"] #non-convex
    #filename=["1695713998-721557","1695714190-6358345","1695714276-641905","1695715304-5775893","1695715518-76627"] #acopf case abundont
    #filename=["1695735578-3853","1695738735-2289534","1695738745-973685"] #acopf case used
    #dropout=0.00,0.05,0.10,0.15,0.20,0.3 
    #filename=["1695818060-0492225","1695818349-963277","1695818645-7361343","1695818832-7737434","1695819052-9986932","1695819080-3326921"] #50*50
    #filename=["1695821069-1573565","1695821224-5350223","1695821246-3657088","1695821261-0735006","1695821279-8416398","1695821301-3447523"] #50*30


    choose_index=0
    save_dir = os.path.join('results', str(data), 'nettest', my_hash(str(sorted(list(args.items())))),filename[choose_index])
    extra_dir = os.path.join('results', str(data), 'method', my_hash(str(sorted(list(args.items())))),filename[choose_index])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #arr = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,1])
    hyper_arg = {}
    hyper_arg['hyper_arr'] = np.arange(400) #100或200
    hyper_arg['hyper_message'] = 'frozen_eval'
    beg= 299
    #eval_theta=torch.tensor([0.1,0.3,0.55,0.7,0.9],dtype=torch.float64,device=DEVICE)
    eval_theta=torch.tensor([5.5],dtype=torch.float64,device=DEVICE)
    for k in range(1): 
        exp_stra = default_args.stratage_choose_args_testnet('dc3_wander')
        #network_root=os.path.join(extra_dir,'stratage{}_{}'.format(5,1), 'solver_{}.0_net.dict'.format(beg))
        network_root=os.path.join(extra_dir,'stratage{}_{}'.format(5,1), 'solver_{}_net.dict'.format(beg))
        my_train_net(network_root,data, args, save_dir, k+1, hyper_arg,exp_stra,eval_theta)


def my_train_net(network_root,data, args, save_dir, exp_time, hyper_arg,exp_stra,eval_theta):
    solver_step = 0.001 #args['lr']
    nepochs = 150
    stratage_num = exp_stra["stratage_num"]
    is_corr_evaltrain = exp_stra['is_corr_evaltrain'] 
    is_other_corr = exp_stra['is_other_corr'] 
    is_function_adjust = exp_stra['is_function_adjust'] 
    is_eval_withtrainingdata = exp_stra['is_eval_withtrainingdata']
    is_store_net = exp_stra['is_store_net']
    net_store_step = exp_stra['net_store_step']
    mywanderstep = 2
    frozen_step = 0
    
    border_shreshold = [-0.05,-0.025,-0.01,-0.005,-0.003]
    border_num = 5
    
    batch_size = 64 #args['batchSize']
    num_steps = args['corrTrainSteps'] 


    #Xvalid = data.validX.to(DEVICE)
    Xvalid = data.testX.to(DEVICE)
    train_dataset = TensorDataset(data.trainX)
 
    valid_dataset = TensorDataset(Xvalid)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)
    

    solver_net = NNSolver(data, args)
    solver_net.load_state_dict(torch.load(network_root))
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(solver_opt,milestones=[50,120],gamma=0.1)
    
    #adapt lr
    lip_constant=0.5/data.get_lip()
    lip_vec = lip_constant.repeat(mywanderstep).detach()
    #lip_vec = lip_constant.repeat(mywanderstep,data.ydim-data.neq).detach()
    #lip_vec = lip_constant.repeat(data.ydim-data.neq).detach()
    #active_lr = torch.tensor(lip_constant,dtype=torch.float64,device=DEVICE,requires_grad=True)
    #active_lr = torch.tensor(lip_constant,dtype=torch.float64,device=DEVICE,requires_grad=True).repeat(mywanderstep)#0.0001
    active_lr=torch.tensor(lip_vec,dtype=torch.float64,device=DEVICE,requires_grad=True)
    active_learning_init = torch.tensor(1e-10)
    active_learning_init1 = torch.tensor(1e-10)

    if args['probType']=="acopf57":
        y_groundtruth=0
    else:
        y_groundtruth=gen_ans(data)
        y_groundtruth= torch.tensor(y_groundtruth).cuda(DEVICE)
    stats = {}
    str_arr = saved_str_arr(exp_stra,args)
    cc_step = 0
    
    savedirpath = os.path.join(save_dir, "stratage{}_{}".format(stratage_num,exp_time))
    if not os.path.exists(savedirpath):
        os.makedirs(savedirpath)
    for i in range(nepochs):
        epoch_stats = {}
        # Get valid loss
        solver_net.eval()  
       
        if i>= frozen_step:
            eval_net_ad(data, Xvalid, solver_net, args, 'valid', epoch_stats, exp_stra,i,border_shreshold,active_lr,y_groundtruth,mywanderstep)
        else:
            eval_net(data, Xvalid, solver_net, args, 'valid_{}'.format(exp_stra['case']), epoch_stats)
        # Get train loss
        solver_net.train()
        train_loss = 0
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            Yhat_train = solver_net(Xtrain)
            Ynew_train_arr = grad_steps_correct(data, Xtrain, Yhat_train, args)
            Ynew_train_arr = Ynew_train_arr.cuda(DEVICE)
            step_point = Ynew_train_arr[-1,:,:]
            if i >= frozen_step:
                for kstep in range(mywanderstep):
                    step_point = data.wander_step_mod3(Xtrain,step_point, border_shreshold[kstep],border_num,active_lr[kstep])   
                train_loss = total_loss_arrY2big(data, Xtrain, step_point, Ynew_train_arr, args,eval_theta[exp_time-1])
            else:
                train_loss = total_loss_arrY(data, Xtrain, Ynew_train_arr, args)
            train_loss.sum().backward()   
            #train_loss.sum().backward(retain_graph=True)
            solver_opt.step()
            if active_lr.grad is not None:
                with torch.no_grad():
                    active_lr -= active_lr.grad * active_learning_init
                    #active_lr[active_lr <= 0] = 0
                    #if cc_step%60==1:
                    #    print("step 1 learning rate:",active_lr[0]," step 2 learning rate:",active_lr[1])
                    #    print("step 1 grad:",active_lr.grad[0],"step 2 grad:",active_lr.grad[1]) 
                    active_lr.grad.data.zero_()  
                    cc_step = cc_step+1
                    #print("ready!")
            
            """ if active_lr.grad is not None:
                with torch.no_grad():
                    torch.clamp 
                    active_lr[0,:] -= active_lr.grad[0,:] * active_learning_init
                    active_lr[1,:] -= active_lr.grad[1,:] * active_learning_init1
                    active_lr[active_lr <= 0] = 0

                    if cc_step%60==1:
                        print("step 1 learning rate:",torch.mean(active_lr[0,:],axis=0)," step 2 learning rate:",torch.mean(active_lr[1,:],axis=0))
                        print("step 1 grad:",torch.mean(torch.abs(active_lr.grad[0,:]),axis=0),"step 2 grad:",torch.mean(torch.abs(active_lr.grad[1,:]),axis=0))
                    active_lr.grad.data.zero_()  
                    cc_step = cc_step+1
                    #print("ready!") """
            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            #dict_agg(epoch_stats, 'train_time', train_time, op='sum')
        scheduler2.step()
        """  if i%10==1:
            print("learning rate is: 2.5") """
        trainingmessageprint(i,epoch_stats,exp_stra,frozen_step)
        if not is_other_corr:  
            if i == 0:
                for key in str_arr:
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in str_arr:
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        if i%net_store_step==0 and is_store_net:
            with open(os.path.join(savedirpath, 'solver_{}_net.dict'.format(i/net_store_step)), 'wb') as f:
                torch.save(solver_net.state_dict(), f)
    with open(os.path.join(savedirpath, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    exp_inf_write(savedirpath,hyper_arg,exp_time,exp_stra)
    return solver_net, stats


def corr_eval(data, X, Ynew_arr, prefix, stats,y_groundtruth):
    make_prefix = lambda x: "{}_{}".format(prefix, x)
    stepnum = Ynew_arr.shape[0]
    for i in range(stepnum):
        a1=torch.norm(Ynew_arr[i]-y_groundtruth,dim=1,p=2).pow(2).mean()
        a2=torch.norm(Ynew_arr[i],dim=1,p=2).pow(2).mean()
        #print(10*torch.log10(a1/a2))
        #print(np.mean(data.obj_fn(Ynew_arr[i]).detach().cpu().numpy()))
        dict_agg(stats, make_prefix('objfn_{}'.format(i)), data.obj_fn(Ynew_arr[i]).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_mean_{}'.format(i)), torch.mean(data.ineq_dist(X,Ynew_arr[i]), dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_max_{}'.format(i)), torch.max(data.ineq_dist(X,Ynew_arr[i]), dim=1)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_max_{}'.format(i)), torch.max(torch.abs(data.eq_resid(X, Ynew_arr[i])), dim=1)[0].detach().cpu().numpy())
def grad_steps_correct(data, X, Y, args):
    take_grad_steps = args['useTrainCorr']
    num_steps = 5 #args['corrTrainSteps']
    Y_arr = torch.zeros((num_steps,Y.shape[0],Y.shape[1]))  
    if take_grad_steps:
        lr = args['corrLr']
        momentum =  args['corrMomentum']*0.6
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

def total_loss_arrY(data, X, Y, args):
    #print('total_loss in method')
    obj_cost = data.obj_fn(Y[-1,:,:])     
    ineq_dist = data.ineq_dist(X, Y[-1,:,:]) 
    ineq_cost = torch.norm(ineq_dist, dim=1)  
    eq_cost = torch.norm(data.eq_resid(X, Y[-1,:,:]), dim=1) 
    return obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost + \
            args['softWeight'] * args['softWeightEqFrac'] * eq_cost
    #return torch.norm(Y_beg - Y[-1,:,:],dim=1)

def total_loss_arrY2big(data, X, stepP, Y, args,eval_theta):
    #print('total_loss in method')
    obj_cost2 = data.obj_fn(stepP)     
    ineq_dist2 = data.ineq_dist(X, stepP) 
    ineq_cost2 = torch.norm(ineq_dist2, dim=1)  
    eq_cost2 = torch.norm(data.eq_resid(X, stepP), dim=1) 
    
    obj_cost1 = data.obj_fn(Y[-1,:,:])     
    ineq_dist1 = data.ineq_dist(X,Y[-1,:,:]) 
    ineq_cost1 = torch.norm(ineq_dist1, dim=1)  
    eq_cost1 = torch.norm(data.eq_resid(X, Y[-1,:,:]), dim=1) 
    return 0.5*(obj_cost1 + 0.4* args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost1) + \
            args['softWeight'] * args['softWeightEqFrac'] * eq_cost1 + obj_cost2 + eval_theta * ineq_cost2
    #return torch.norm(Y_beg - Y[-1,:,:],dim=1)

def eval_net(data, X, solver_net, args, prefix, stats):
    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Y = solver_net(X)
    base_end_time = time.time()
    
    Ynew_train_arr = grad_steps_correct(data, X, Y, args)
    Ynew_train_arr = Ynew_train_arr.cuda(DEVICE)

    Ycorr = Ynew_train_arr[-1,:,:]
    end_time = time.time()
    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Ycorr, args).detach().cpu().numpy()) 
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy())
    dict_agg(stats, make_prefix('dist'), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'), torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy())
    return stats  

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
        return ineq_effective * obj_cost +  15*(1 - args['softWeightEqFrac']) * ineq_cost + \
            args['softWeightEqFrac'] * eq_cost
    else:
        return obj_cost + ineq_effective * (1 - args['softWeightEqFrac']) * ineq_cost + \
            args['softWeightEqFrac'] * eq_cost


@torch.no_grad()
def loss_effective_gen(based_grad,adaptive_grad): 
    is_exp = True
    based_norm = torch.norm(based_grad,p=2,dim=1).view(-1,1) + 1e-6 
    base_stand = torch.div(based_grad,based_norm)
    adaptive_norm = torch.norm(adaptive_grad,p=2,dim=1).view(-1,1) + 1e-6  
    adaptive_stand = torch.div(adaptive_grad,adaptive_norm) 
    effective_grad = torch.mul(base_stand,adaptive_stand).sum(axis=1)
    if is_exp:
        res_effective = torch.exp(effective_grad+1) 
    return res_effective

@torch.no_grad()
def corr_ineq_wander(data,X,Y,ineq_diff,ineq_sign,fn_diff,lr_corr):
    border_shreshold = 1e-6 
    ineq_shreshold = -1e-3
    border_sign_bool = torch.zeros_like(ineq_sign)
    border_sign_bool[ineq_sign>ineq_shreshold] = 2

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
    #str.append('valid_{}_steps'.format(case))
    str.append('valid_{}_time'.format(case))
    return str

def trainingmessageprint(i,epoch_stats,exp_stra,frozestep):
    #is_other_corrtrain = exp_stra['is_other_corrtrain']
    case = exp_stra['case']
    step = 5 #####defaults['corrTestMaxSteps'] = 10
    is_other_corr =exp_stra['is_other_corr']
    is_corr_evaltrain = exp_stra['is_corr_evaltrain']
    if is_corr_evaltrain:
        print(
            'Epoch {}:  fnbeg {:.4f}, fnbeg2 {:.4f}, fnbeg3 {:.4f}, fnqend1 {:.4f},fnend2 {:.4f}'.format(
                i, np.mean(epoch_stats['train_{}_objfn_{}'.format(case,0)]),
                np.mean(epoch_stats['train_{}_objfn_{}'.format(case,1)]),np.mean(epoch_stats['train_{}_objfn_{}'.format(case,2)]),
                np.mean(epoch_stats['train_{}_objfn_{}'.format(case,step-2)]),np.mean(epoch_stats['train_{}_objfn_{}'.format(case,step-1)])))
        print(
            'Epoch {}:  ineqbeg {:.4f}, ineqbeg2 {:.4f}, beg3 {:.4f}, end1 {:.4f},end2 {:.4f}'.format(
                i, np.mean(epoch_stats['train_{}_ineq_mean_{}'.format(case,0)]),
                np.mean(epoch_stats['train_{}_ineq_mean_{}'.format(case,1)]),np.mean(epoch_stats['train_{}_ineq_mean_{}'.format(case,2)]),
                np.mean(epoch_stats['train_{}_ineq_mean_{}'.format(case,step-2)]),np.mean(epoch_stats['train_{}_ineq_mean_{}'.format(case,step-1)])))
    else:
        if i >= frozestep:
            step = 3 #exp_stra['step']
            print("epoch is {}".format(i))
            for k in range(step):
                print(
                    'iter {}: train loss {:.4f}, fnobj {:.4f}, ineqmax {:.6f}, eqmax{:.8f}'.format(
                        k, np.mean(epoch_stats['train_loss']),
                        np.mean(epoch_stats['valid_{}_objfn_{}'.format(case,k)]),np.mean(epoch_stats['valid_{}_ineq_max_{}'.format(case,k)]),
                        np.mean(epoch_stats['valid_{}_eq_max_{}'.format(case,k)])))
                wandb.log({"train loss":epoch_stats['train_loss'],"fn_obj_step{}".format(k):np.mean(epoch_stats['valid_{}_objfn_{}'.format(case,k)]),
                          "valid_{}_ineq_max".format(k): np.mean(epoch_stats['valid_{}_ineq_max_{}'.format(case,k)]),
                          "valid_{}_eq_max".format(k): np.mean(epoch_stats['valid_{}_eq_max_{}'.format(case,k)]) })
        else:
            print(
                'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, eq max {:.4f}, time {:.4f}'.format(
                    i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_{}_eval'.format(case)]),
                    np.mean(epoch_stats['valid_{}_ineq_max'.format(case)]), np.mean(epoch_stats['valid_{}_ineq_mean'.format(case)]), 
                    np.mean(epoch_stats['valid_{}_eq_max'.format(case)]),np.mean(epoch_stats['valid_{}_time'.format(case)])))

@torch.no_grad()
def eval_net_ad(data, X, solver_net, args, prefix, states,exp_stra,nepochs,border_shreshold,active_lr,y_groundtruth,mywanderstep): 
    is_timeprint = True  
    eps_converge = args['corrEps']
    lr_corr = 0.001 #exp_stra['lr_corr']
    step = mywanderstep #exp_stra['step']
    case = exp_stra['case']
    #border_shreshold = [-0.05,-0.03,-0.02,-0.015,-0.010,-0.005,-0.005,-0.003,-0.002,-0.001,-0.001,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2]
    border_num = 5 #or 7
    iter_num = border_num

    start_time = time.time()
    Y = solver_net(X)
    base_end_time = time.time()
    Ynew_train_arr = grad_steps_correct(data, X, Y, args)
    base_end_time1 = time.time()
    Ynew_train_arr = Ynew_train_arr.cuda(DEVICE)

    Ycorr = Ynew_train_arr[-1,:,:]
    end_time = time.time()
    if nepochs>=0:
        is_wander_or_ad = exp_stra['is_wander_or_ad']
        if is_wander_or_ad:
            newapp = True #jiae
            newthirdversion = True #jiae
            if newapp:
                if newthirdversion:
                    Ycorr_arr = torch.zeros((step+1,Ycorr.shape[0],Ycorr.shape[1]),device=DEVICE)
                    Ycorr_arr[0,:,:]=Ycorr
                    for i in range(step):
                        #Ycorr_arr[i+1,:,:] = data.wander_step_mod3(X, Ycorr_arr[i,:,:], border_shreshold[1],border_num,active_lr[i])
                        Ycorr_arr[i+1,:,:] = data.wander_step_mod3(X, Ycorr_arr[i,:,:], border_shreshold[i],border_num,active_lr[i])  
                    end_time1 = time.time()  
                    #print("time is:", end_time1-start_time)   
                    corr_eval(data,X,Ycorr_arr,'{}_{}'.format(prefix,case),states,y_groundtruth)
                else:
                    Ycorr_arr = torch.zeros((step+1,Ycorr.shape[0],Ycorr.shape[1]),device=DEVICE)
                    Ycorr_arr[0,:,:]=Ycorr
                    for i in range(step):
                        border_index = data.getborderindex(X,Y,border_shreshold,border_num)
                        Ycorr_arr[i+1,:,:] = data.wander_step_modified(X, Y, border_index,border_num,iter_num)                    
                    corr_eval(data,X,Ycorr_arr,'{}_{}'.format(prefix,case),states,y_groundtruth)
            else:
                #Ycorr_arr = torch.zeros((step+1,Ycorr.shape[0],Ycorr.shape[1]))
                Ycorr_arr = torch.zeros((step+1,Ycorr.shape[0],Ycorr.shape[1]),device=DEVICE)
                Ycorr_arr[0,:,:]=Ycorr
                for i in range(step):
                    ineq_diff = data.ineq_diff_get()
                    fn_diff = data.fn_diff_get(Ycorr_arr[i])
                    ineq_sign = data.ineq_sign_gen(X,Ycorr_arr[i])
                    Ycorr_arr[i+1,:,:] = corr_ineq_wander(data,X,Ycorr_arr[i],ineq_diff,ineq_sign,fn_diff,lr_corr)
                corr_eval(data,X,Ycorr_arr,'{}_{}'.format(prefix,case),states)
        else:
            #Ycorr_arr = torch.zeros((step+1,Ycorr.shape[0],Ycorr.shape[1]))
            Ycorr_arr = torch.zeros((step+1,Ycorr.shape[0],Ycorr.shape[1]),device=DEVICE)
            Ycorr_arr[0,:,:]=Ycorr
            for i in range(step):
                ineq_diff = data.ineq_diff_get()
                fn_diff = data.fn_diff_get(Ycorr_arr[i])
                ineq_sign = data.ineq_sign_gen(X,Ycorr_arr[i])
                #Ycorr_arr[i+1,:,:] = corr_gradient_gen_sum(data,X,Ycorr_arr[i],ineq_diff,ineq_sign,fn_diff,lr_corr)
            corr_eval(data,X,Ycorr_arr,'{}_{}'.format(prefix,case), states)
    else:
        Ycorr_arr = torch.zeros((1+step,Ycorr.shape[0],Ycorr.shape[1]),device=DEVICE)
        for steptime in range(step+1):
            Ycorr_arr[steptime,:,:] =Ycorr
        corr_eval(data,X,Ycorr_arr,'{}_{}'.format(prefix,case),states)
    base_end_time2 = time.time()
    print("Net solve time:{}, Correct time:{}, Active time:{}".format(base_end_time-start_time,base_end_time1-base_end_time,base_end_time2-base_end_time1))
@torch.no_grad()     
def gen_ans(data):
    valid_dataset = TensorDataset(data.validX)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)
    
    P = np.array(data.Q.to("cpu"))
    #P = torch.randn(100,100)
    q = np.array(data.p.to("cpu"))
    A = np.array(data.A.to("cpu"))
    
    G = np.array(data.G.to("cpu"))
    h = np.array(data.h.to("cpu"))
    solver ="osqp"
    datanum = len(valid_dataset)
    x = np.zeros((datanum,100))
    Xvalid = data.validX
    for i in range(datanum):
        b = np.array(Xvalid[i,:].to("cpu"))
        x[i,:] = solve_qp(P, q, G, h, A, b, solver=solver)
    return x
            
    
           
if __name__=='__main__':
    main()
   