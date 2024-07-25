import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float64)

import numpy as np
import osqp
from qpth.qp import QPFunction

from scipy.linalg import svd
from scipy.sparse import csc_matrix

import hashlib
from copy import deepcopy
import scipy.io as spio
import time
from pypower.api import case57
from pypower.api import opf, makeYbus
from pypower import idx_bus, idx_gen, ppoption

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{value} is not a valid boolean value')

def my_hash(string):
    return hashlib.sha1(bytes(string, 'utf-8')).hexdigest()


###################################################################
# SIMPLE PROBLEM
###################################################################

class SimpleProblem:
    """ 
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
    """
    def __init__(self, Q, p, A, G, h, X, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._G = torch.tensor(G)
        self._h = torch.tensor(h)
        self._X = torch.tensor(X)
        self._Y = None
        self._xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = G.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        det = 0
        i = 0
        while abs(det) < 0.000001 and i < 100: # abs(det) < 0.0001 and i < 100:
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
            det = torch.det(self._A[:, self._other_vars]) #Prevent linear dependence.
            i += 1
        if i == 100:
            raise Exception
        else:
            self._A_partial = self._A[:, self._partial_vars]
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])

        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'SimpleProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def h(self):
        return self._h

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num*self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def obj_fn(self, Y):
        return (0.5*(Y@self.Q)*Y + self.p*Y).sum(dim=1)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_dist@self.G
    
    @torch.no_grad()
    def get_lip(self):
        Q_half = self.Q[:, self.partial_vars] - self.Q[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        com_tran_Q = Q_half.T[:, self.partial_vars] - Q_half.T[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        value,_ = torch.linalg.eig(com_tran_Q)
        return torch.max(value[0].real)

    #obtain the gradient after correction 
    def ineq_partial_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) #the new coefficient G after completion 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  #the new coefficient h after completion 
        grad = 2 * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective 
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y
    
    def newton_corr_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T 
        G_effectivesum = torch.norm(G_effective,dim=1,p=2)
        grad = torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ (torch.div(G_effective,G_effectivesum.view(-1,1))/2500)
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] =  grad
        Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y
    
    @torch.no_grad()
    def fn_diff_get(self,Y):
        fn_grad = (0.5*Y @(self.Q+self.Q.T))+self.p #vector of 1*Ydim
        afterdealeq_diff = fn_grad[:,self.partial_vars] - fn_grad[:,self.other_vars]  @  (self._A_other_inv @ self._A_partial)
        #cmp_fndiff = torch.zeros(Y.shape[0], self.ydim, device=self.device)
        #cmp_fndiff[:, self.partial_vars] =  afterdealeq_diff
        #cmp_fndiff[:, self.other_vars] = - (afterdealeq_diff @ self._A_partial.T) @ self._A_other_inv.T
        #return cmp_fndiff
        return afterdealeq_diff
    
    @torch.no_grad()   
    def ineq_diff_get(self):
        return self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
    
    
    @torch.no_grad() 
    def getborderindex(self,X,Y,border_shreshold,border_num):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  
        border_value = Y[:, self.partial_vars] @ G_effective.T - h_effective #number of sample *number of ineq
        border_sortedvalue, bordersorted_index = torch.sort(border_value,descending = True,dim = -1)
        NAN_value_num = border_value.shape[1] #number of ineq
        final_index=torch.zeros((border_value.shape[0],border_num),dtype=torch.int64,device=DEVICE)
        for i in range(border_value.shape[0]):
            tmp_index= bordersorted_index[i,:][border_sortedvalue[i,:]>=border_shreshold]
            inequality_index = torch.ones(border_value.shape[1],dtype=torch.int64,device=DEVICE)*NAN_value_num
            inequality_index[0:min(border_num,tmp_index.shape[0])] = tmp_index[0:min(border_num,tmp_index.shape[0])]
            final_index[i,:] = inequality_index[0:border_num]
        return final_index
              
    @torch.no_grad()
    def lr_gen1(self, drt_vec, Y, border_index,satisified_num):
       min_value = 1e-6
       fn_lip = 0.2
       new_h = self.h - Y @ self.G.T
       #print("befor:",new_h[0:10,3])
       new_h[new_h<=min_value] = min_value
       #print("after:",new_h[0:10,3])
       grad = - drt_vec @ self.G.T
       grad[grad<min_value] = min_value
       tmp=torch.div(new_h,grad)
       for i in range(border_index.shape[0]):
           kk_tmp=border_index[i,0:int(satisified_num[i])]
           bool_vec = [kk_tmp<self._nineq]
           tmp_index=border_index[i,0:int(satisified_num[i])][bool_vec]
           tmp[i,tmp_index]= 0.2
       #print(grad[0:10,3])
       min_vec,index = torch.min(tmp,dim=1)
       min_vec[min_vec>=fn_lip]=fn_lip
       return  min_vec
    
    def lr_gen(self, drt_vec, Y):
       min_value = 1e-6
       fn_lip = 0.2
       new_h = self.h - Y @ self.G.T
       new_h[new_h<=min_value] = min_value
       grad = - drt_vec @ self.G.T
       grad[grad<min_value] = min_value
       tmp=torch.div(new_h,grad)
       min_vec,index = torch.min(tmp,dim=1)
       min_vec[min_vec>=fn_lip]=fn_lip
       return  min_vec
        
    def wander_step_mod3(self, X, Y,border_shreshold,border_num,active_lr):
        mytestsign = True
        datanum = Y.shape[0]
        zero_shreshold = 1e-7
        ineq_num = self._nineq #number of inequation
        eq_num = self._neq #number of equation
        dim_num = self.ydim #dimension of y
        feasible_num = dim_num - eq_num
        #position swap for random selection(simulate random)
        random_pick = np.random.choice(feasible_num, feasible_num, replace=False)
        random_pos =self.partial_vars[random_pick]
        #generate the boundary condition
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  
        border_value = Y[:, self.partial_vars] @ G_effective.T - h_effective 
        border_sortedvalue, bordersorted_index = torch.sort(border_value,descending = True,dim = -1)
        border_index = bordersorted_index[:,:border_num]
        border_index[border_sortedvalue[:,:border_num]<border_shreshold] =  ineq_num
        #innitialize the gradient
        fn_grad = self.fn_diff_get(Y)[:,random_pick]
        #construct the boundary inequality matrix
        ineq_tmp= torch.zeros((ineq_num + 1, feasible_num), device=self.device)
        ineq_tmp[0:-1, :] = G_effective[:,random_pick]
        ineq_tmp[-1, :] = 0
        ineq_matrix = ineq_tmp[ border_index,:]
        #initialize the signal
        satisified_sign = torch.ones(datanum,dtype=torch.int32,device=self.device) #determine whether the algorithm has terminated: 1 for "cannot terminate" and 0 for "can terminate."
        satisified_num = torch.zeros(datanum,dtype=torch.int32,device=self.device) #record in which iteration BWS finished
        zero_sign = torch.zeros(datanum,dtype=torch.int32,device=self.device) #Check if 0 is detected; if detected, skip it.
        for iter in range(border_num):
            #form equivalent problems through slicing
            if ~torch.any(satisified_sign.bool()):
                break
            sub_ineq = ineq_matrix[satisified_sign.bool(),:,:]
            sub_fn = fn_grad[satisified_sign.bool(),:]
            #determine if the conditions are satisfied
            diff_minval, diff_minindex = torch.min(torch.mul(sub_fn[:,iter:].view(-1,1,feasible_num-iter),sub_ineq[:,iter:,iter:]).sum(dim=2),dim=1)
            subsatisify_sign = diff_minval >= 0
            subsatisify_inv = subsatisify_sign == False
            #Update the inequalities
            #Prevent linear dependence among inequalities matrix
            divzerosign_bool = torch.abs(sub_ineq[:,iter,iter]) <= zero_shreshold
            divnozerosign_bool = (divzerosign_bool==False)
            if ~torch.any(divnozerosign_bool):
                fn_grad[satisified_sign.bool(),:]=0
                break 
            #handle inequalities that do not start close to 0.
            sub_ineq[divnozerosign_bool,iter+1:,iter+1:] = sub_ineq[divnozerosign_bool,iter+1:,iter+1:] - \
                torch.mul(sub_ineq[divnozerosign_bool,iter:iter+1,iter+1:],torch.div(sub_ineq[divnozerosign_bool,iter+1:,iter:iter+1],sub_ineq[divnozerosign_bool,iter:iter+1,iter:iter+1]))
            #handle function gradients that do not start close to 0.
            sub_fn[divnozerosign_bool,iter+1:] = sub_fn[divnozerosign_bool,iter+1:] - \
                torch.mul(sub_ineq[divnozerosign_bool,iter,iter+1:],torch.div(sub_fn[divnozerosign_bool,iter:iter+1],sub_ineq[divnozerosign_bool,iter:iter+1,iter]))

            #signal update
            test_sign = (divzerosign_bool & subsatisify_inv) 
            if test_sign.numel():
                zero_sign[satisified_sign.bool()] = test_sign.int()
            #Update the global state based on the sign
            satisified_sign[satisified_sign.bool()] = subsatisify_inv.int() 
            #update the global function and inequalities
            ineq_matrix[satisified_sign.bool(),:,:] = sub_ineq[subsatisify_inv,:,:] 
            fn_grad[satisified_sign.bool(),:] = sub_fn[subsatisify_inv,:]
            #update the global state 
            signofzero = satisified_sign.bool() & zero_sign.bool() 
            if torch.any(signofzero):
                satisified_sign[signofzero] = 0 
            satisified_num = satisified_num + satisified_sign
        #recover process
        for iter in range(border_num):
            deal_sign = satisified_num >= (border_num-iter)
            if torch.any(deal_sign):
                fn_grad[deal_sign,border_num-iter-1] = torch.div(-torch.mul(fn_grad[deal_sign,border_num-iter:], \
                    ineq_matrix[deal_sign, border_num-iter-1, border_num-iter:]).sum(dim=1),ineq_matrix[deal_sign, border_num-iter-1, border_num-iter-1])
        #drt_grad = torch.zeros(datanum,dim_num,device=self.device)
        #drt_grad[:, random_pos] = fn_grad
        #drt_grad[:, self.other_vars] = - ( drt_grad[:, self.partial_vars] @ self._A_partial.T) @ self._A_other_inv.T
        #generate the lr_rate 
        min_value = -1e-6
        div_minvalue = 1e-8
        border_value[border_value>=min_value] = min_value
        
       
        indeedgrad = torch.zeros_like(fn_grad)
        indeedgrad[:,random_pick] = fn_grad
    
        grad_multi = -indeedgrad @ G_effective.T
        #prevent division by zero.
        grad_multi[grad_multi<div_minvalue] = div_minvalue
        tmp=torch.div(-border_value,grad_multi)
        #print(grad[0:10,3])
        min_vec,index = torch.min(tmp,dim=1)
    
        min_vec[ min_vec>=active_lr] = active_lr 
        tmp_fn = torch.mul(indeedgrad, min_vec.view(-1,1))
            
        drt_grad = torch.zeros(datanum,dim_num,device=self.device)    
        drt_grad[:, self.partial_vars] = tmp_fn
        drt_grad[:, self.other_vars] = - ( tmp_fn @ self._A_partial.T) @ self._A_other_inv.T
        return  Y - drt_grad
        
    @torch.no_grad()
    def ineq_partial_grad_nograd(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  
        grad = 2 * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective 
        Y_step = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y_step[:, self.partial_vars] = grad
        Y_step[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y_step
    
    #d(fn)/d(y'), here, y' represents the free variables after completion
    @torch.no_grad()
    def fn_partial_grad_nograd(self, X, Y):
        fn_diff_Ycom = self.p + 0.5 * Y @ (self.Q.T+self.Q)
        fn_diff_Y = fn_diff_Ycom[:,self.partial_vars] + fn_diff_Ycom[:,self.other_vars] @ (self._A_other_inv @ self._A_partial)
        fn_grad = torch.zeros(X.shape[0], self.ydim, device=self.device)
        fn_grad[:, self.partial_vars] = fn_diff_Y
        fn_grad[:, self.other_vars] = - (fn_diff_Y @ self._A_partial.T) @ self._A_other_inv.T
        return fn_grad

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y


    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)  
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y
    
    @torch.no_grad()
    def complete_stepgenY(self,X,Y,step,lr):
        tmp = Y[:,self.partial_vars] 
        newY_partial = step * lr + tmp
        return self.complete_partial(X,newY_partial)
    
    def opt_solve(self, X, solver_type='osqp', tol=1e-4):

        if solver_type == 'qpth':
            print('running qpth')
            start_time = time.time()
            res = QPFunction(eps=tol, verbose=False)(self.Q, self.p, self.G, self.h, self.A, X)
            end_time = time.time()

            sols = np.array(res.detach().cpu().numpy())
            total_time = end_time - start_time
            parallel_time = total_time
        
        elif solver_type == 'osqp':
            print('running osqp')
            Q, p, A, G, h = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            for Xi in X_np:
                solver = osqp.OSQP()
                my_A = np.vstack([A, G])
                my_l = np.hstack([Xi, -np.ones(h.shape[0]) * np.inf])
                my_u = np.hstack([Xi, h])
                solver.setup(P=csc_matrix(Q), q=p, A=csc_matrix(my_A), l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += (end_time - start_time)
                if results.info.status == 'solved':
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time/len(X_np)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time


###################################################################
# NONCONVEX PROBLEM
###################################################################

class NonconvexProblem:
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
    """
    def __init__(self, Q, p, A, G, h, X, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._G = torch.tensor(G)
        self._h = torch.tensor(h)
        self._X = torch.tensor(X)
        self._Y = None
        self._xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = G.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        det = 0
        i = 0
        while abs(det) < 0.0001 and i < 100:
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
            det = torch.det(self._A[:, self._other_vars])
            i += 1
        if i == 100:
            raise Exception
        else:
            self._A_partial = self._A[:, self._partial_vars]
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])
            self._M = 2 * (self.G[:, self.partial_vars] -
                            self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial))

        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'NonconvexProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def h(self):
        return self._h

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num*self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def obj_fn(self, Y):
        return (0.5*(Y@self.Q)*Y + self.p*torch.sin(Y)).sum(dim=1)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        return 2 * torch.clamp(Y@self.G.T - self.h, 0) @ self.G

    def ineq_partial_grad(self, X, Y):
        grad = torch.clamp(Y@self.G.T - self.h, 0) @ self._M
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    @torch.no_grad()
    def get_lip(self):
        Q_half = self.Q[:, self.partial_vars] - self.Q[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        com_tran_Q = Q_half.T[:, self.partial_vars] - Q_half.T[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        value,_ = torch.linalg.eig(com_tran_Q)
        return torch.max(value[0].real)
    
    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    #Obtain the current function's differential.
    @torch.no_grad()
    def fn_diff_get(self,Y):
        fn_grad = (0.5*Y @(self.Q+self.Q.T))+self.p * torch.cos(Y)
        afterdealeq_diff = fn_grad[:,self.partial_vars] - fn_grad[:,self.other_vars] @  (self._A_other_inv @ self._A_partial)
        return afterdealeq_diff
    
    @torch.no_grad()   
    def ineq_diff_get(self):
        return self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
    
    @torch.no_grad()    
    def wander_step_gen(self, X, Y, sign,lr):
        datanum = sign.shape[0]
        
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  
        newY = torch.zeros(X.shape[0], self.ydim, device=self.device)
        index_sign = torch.arange(0,sign.shape[1],1)
        
        for i in range(datanum):
            bool_ineqcomplete = index_sign[sign[i,:]]
            eqineqn = torch.ones_like(bool_ineqcomplete).numel() 
            new_G = G_effective[bool_ineqcomplete]
            new_H = h_effective[i,bool_ineqcomplete]
            det = 0
            t = 0
            if eqineqn==0:
                fn_grad = (0.5*Y[i,:]@(self.Q+self.Q.T))+self.p* torch.cos(Y)
                afterdealeq_diff = fn_grad[self.partial_vars] - fn_grad[self.other_vars] @  (self._A_other_inv @ self._A_partial) 
                Ystep = torch.zeros((self.ydim), device=self.device)
                Ystep[self.partial_vars] = afterdealeq_diff
                Ystep[self.other_vars] = -(afterdealeq_diff @ self._A_partial.T) @ self._A_other_inv.T 
                Ystep = Ystep/ torch.norm(Ystep,dim=0) 
                newY[i,:] = -Ystep * lr +Y[i,:]
            else:
                while abs(det) < 0.0001 and t < 20:
                    partial_vars = np.random.choice(self._ydim - self._neq, self._ydim - self._neq - eqineqn  , replace=False)
                    other_vars = np.setdiff1d( np.arange(self._ydim - self._neq), partial_vars)
                    det = torch.det(new_G[:, other_vars]) 
                    t += 1
                if t == 100:
                    raise Exception
                else:
                    G_partial = new_G[:, partial_vars]
                    G_other_inv = torch.inverse(new_G[:, other_vars])
                    #NG_effective = new_G[:, partial_vars] - new_G[:, other_vars] @ (G_other_inv @ G_partial)
                    fn_grad = (0.5*Y[i,:]@(self.Q+self.Q.T))+self.p* torch.cos(Y)
                    afterdealeq_diff = fn_grad[self.partial_vars] - fn_grad[self.other_vars] @  (self._A_other_inv @ self._A_partial) 
                    afterdealineq_diff = afterdealeq_diff[partial_vars] - afterdealeq_diff[other_vars] @ (G_other_inv @ G_partial) 
                    if False: #(deprecated)
                        Ypartial = torch.zeros((self.ydim-self.neq), device=self.device)
                        Ypartial[partial_vars] =  Y[i,self.partial_vars][partial_vars] - lr * afterdealineq_diff #修改现有自由变量
                        Ypartial[other_vars] = (new_H - Ypartial[partial_vars] @ G_partial.T) @ G_other_inv.T
                        newYtmp = torch.zeros((self.ydim), device=self.device)
                        newYtmp[self.other_vars] =  (X[i,:] - Ypartial @ self._A_partial.T) @ self._A_other_inv.T 
                        newYtmp[self.partial_vars] = Ypartial
                        newY[i,:] = newYtmp
                    else: 
                        Ystep = torch.zeros((self.ydim), device=self.device)
                        Ystep_tmp = torch.zeros((self.ydim-self.neq), device=self.device)
                        Ystep_tmp[partial_vars] = afterdealineq_diff
                        Ystep_tmp[other_vars] = -(afterdealineq_diff @ G_partial.T) @ G_other_inv.T
                        Ystep[self.partial_vars] = Ystep_tmp
                        Ystep[self.other_vars] = -(Ystep_tmp @ self._A_partial.T) @ self._A_other_inv.T 
                        Ystep = Ystep/ torch.norm(Ystep,dim=0) 
                        newY[i,:] = -Ystep * lr +Y[i,:]
        return newY
                
    @torch.no_grad()
    def ineq_sign_gen(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T
        ineq_residual = Y[:, self.partial_vars] @ G_effective.T - h_effective
        return  ineq_residual
    
    @torch.no_grad()
    def ineq_partial_grad_nograd(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  
        grad = 2 * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective 
        Y_step = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y_step[:, self.partial_vars] = grad
        Y_step[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y_step
    
    @torch.no_grad()
    def fn_partial_grad_nograd(self, X, Y):
        fn_diff_Ycom = self.p* torch.cos(Y) + 0.5 * Y @ (self.Q.T+self.Q)
        fn_diff_Y = fn_diff_Ycom[:,self.partial_vars] + fn_diff_Ycom[:,self.other_vars] @ (self._A_other_inv @ self._A_partial)
        fn_grad = torch.zeros(X.shape[0], self.ydim, device=self.device)
        fn_grad[:, self.partial_vars] = fn_diff_Y
        fn_grad[:, self.other_vars] = - (fn_diff_Y @ self._A_partial.T) @ self._A_other_inv.T
        return fn_grad
    
    @torch.no_grad() 
    def getborderindex(self,X,Y,border_shreshold,border_num):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  
        border_value = Y[:, self.partial_vars] @ G_effective.T - h_effective 
        border_sortedvalue, bordersorted_index = torch.sort(border_value,descending = True,dim = -1)
        NAN_value_num = border_value.shape[1] 
        final_index=torch.zeros((border_value.shape[0],border_num),dtype=torch.int64,device=DEVICE)
        for i in range(border_value.shape[0]):
            tmp_index= bordersorted_index[i,:][border_sortedvalue[i,:]>=border_shreshold]
            inequality_index = torch.ones(border_value.shape[1],dtype=torch.int64,device=DEVICE)*NAN_value_num
            inequality_index[0:min(border_num,tmp_index.shape[0])] = tmp_index[0:min(border_num,tmp_index.shape[0])]
            final_index[i,:] = inequality_index[0:border_num]
        return final_index

    @torch.no_grad() #a_row=1*n,a_col=n*1
    def fast_inversegen(self,A,a_row,a_col,a,A_inv):
        hyper_zero_shreshold = -1e-6
        datanum = A.shape[0]
        dim = A.shape[1]   
        v_1 = torch.bmm(A_inv, a_col.view(datanum,-1,1)).view(datanum,-1)
        minv_1,minv_1index=torch.min(v_1,dim=1) #???
        maxv_1,maxv_1index=torch.max(v_1,dim=1) #???
        ver_zero = a - torch.mul(a_row,v_1).sum(dim=1)
        relevantsign = torch.abs(ver_zero) >= hyper_zero_shreshold
        fixnum = torch.ones_like(relevantsign)[relevantsign].sum(dim=0)        
        B = torch.zeros((fixnum,dim+1,dim+1),device=self.device)
        B[:,-1,-1] = torch.div(1,ver_zero[relevantsign])
        B[:,0:-1,-1] = -torch.mul(v_1[relevantsign,:],B[:,-1:,-1])
        B_row = -torch.bmm(a_row[relevantsign,:].view(fixnum,1,-1),A_inv[relevantsign,:,:]).view(fixnum,1,-1)
        B[:,-1:,0:-1] =  torch.bmm(B[:,-1,1].view(-1,1,1), B_row)        
        B[:,0:-1,0:-1] = A_inv[relevantsign,:,:]- torch.bmm(v_1[relevantsign,:].view(fixnum,-1,1),B[:,-1:,0:-1])
        return B,relevantsign 
    
    def wander_step_mod3(self, X, Y,border_shreshold,border_num,active_lr):
        mytestsign = True
        datanum = Y.shape[0]
        zero_shreshold = 1e-7
        ineq_num = self._nineq 
        eq_num = self._neq 
        dim_num = self.ydim
        feasible_num = dim_num - eq_num
      
        random_pick = np.random.choice(feasible_num, feasible_num, replace=False)
        random_pos =self.partial_vars[random_pick]
        
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  
        border_value = Y[:, self.partial_vars] @ G_effective.T - h_effective 
        border_sortedvalue, bordersorted_index = torch.sort(border_value,descending = True,dim = -1)
        border_index = bordersorted_index[:,:border_num]
        border_index[border_sortedvalue[:,:border_num]<border_shreshold] =  ineq_num
        
        fn_grad = self.fn_diff_get(Y)[:,random_pick]
       
        ineq_tmp= torch.zeros((ineq_num + 1, feasible_num), device=self.device)
        ineq_tmp[0:-1, :] = G_effective[:,random_pick]
        ineq_tmp[-1, :] = 0
        ineq_matrix = ineq_tmp[ border_index,:]
        
        satisified_sign = torch.ones(datanum,dtype=torch.int32,device=self.device) 
        satisified_num = torch.zeros(datanum,dtype=torch.int32,device=self.device) 
        zero_sign = torch.zeros(datanum,dtype=torch.int32,device=self.device) 
        for iter in range(border_num):
            
            if ~torch.any(satisified_sign.bool()):
                break
            sub_ineq = ineq_matrix[satisified_sign.bool(),:,:]
            sub_fn = fn_grad[satisified_sign.bool(),:]
            
            diff_minval, diff_minindex = torch.min(torch.mul(sub_fn[:,iter:].view(-1,1,feasible_num-iter),sub_ineq[:,iter:,iter:]).sum(dim=2),dim=1)
            subsatisify_sign = diff_minval >= 0
            subsatisify_inv = subsatisify_sign == False
            
            divzerosign_bool = torch.abs(sub_ineq[:,iter,iter]) <= zero_shreshold
            divnozerosign_bool = (divzerosign_bool==False)
            
            sub_ineq[divnozerosign_bool,iter+1:,iter+1:] = sub_ineq[divnozerosign_bool,iter+1:,iter+1:] - \
                torch.mul(sub_ineq[divnozerosign_bool,iter:iter+1,iter+1:],torch.div(sub_ineq[divnozerosign_bool,iter+1:,iter:iter+1],sub_ineq[divnozerosign_bool,iter:iter+1,iter:iter+1]))
            
            sub_fn[divnozerosign_bool,iter+1:] = sub_fn[divnozerosign_bool,iter+1:] - \
                torch.mul(sub_ineq[divnozerosign_bool,iter,iter+1:],torch.div(sub_fn[divnozerosign_bool,iter:iter+1],sub_ineq[divnozerosign_bool,iter:iter+1,iter]))
            
            test_sign = (divzerosign_bool & subsatisify_inv) 
            if test_sign.numel():
                zero_sign[satisified_sign.bool()] = test_sign.int()
            
            satisified_sign[satisified_sign.bool()] = subsatisify_inv.int() 
           
            ineq_matrix[satisified_sign.bool(),:,:] = sub_ineq[subsatisify_inv,:,:] 
            fn_grad[satisified_sign.bool(),:] = sub_fn[subsatisify_inv,:]
            
            signofzero = satisified_sign.bool() & zero_sign.bool() 
            if torch.any(signofzero):
                satisified_sign[signofzero] = 0 
            satisified_num = satisified_num + satisified_sign
        
        for iter in range(border_num):
            deal_sign = satisified_num >= (border_num-iter)
            if torch.any(deal_sign):
                fn_grad[deal_sign,border_num-iter-1] = torch.div(-torch.mul(fn_grad[deal_sign,border_num-iter:], \
                    ineq_matrix[deal_sign, border_num-iter-1, border_num-iter:]).sum(dim=1),ineq_matrix[deal_sign, border_num-iter-1, border_num-iter-1])
        
        #drt_grad = torch.zeros(datanum,dim_num,device=self.device)
        #drt_grad[:, random_pos] = fn_grad
        #drt_grad[:, self.other_vars] = - ( drt_grad[:, self.partial_vars] @ self._A_partial.T) @ self._A_other_inv.T
        #lr_rate
        min_value = -1e-6
        div_minvalue = 1e-8
        border_value[border_value>=min_value] = min_value
        indeedgrad = torch.zeros_like(fn_grad)
        indeedgrad[:,random_pick] = fn_grad
        
        grad_multi = -indeedgrad @ G_effective.T
        
        grad_multi[grad_multi<div_minvalue] = div_minvalue
        tmp=torch.div(-border_value,grad_multi)
        #print(grad[0:10,3])
        min_vec,index = torch.min(tmp,dim=1)
        
        if True:
            min_vec[min_vec>=active_lr] = active_lr
            drt_grad = torch.zeros(datanum,dim_num,device=self.device)
            tmp_fn = torch.mul(indeedgrad, min_vec.view(-1,1))
        else:
            min_matrix = min_vec.repeat()
            min_vec[min_vec>=active_lr] = active_lr
            print("aaa")
            
        drt_grad[:, self.partial_vars] = tmp_fn
        drt_grad[:, self.other_vars] = - ( tmp_fn @ self._A_partial.T) @ self._A_other_inv.T
        return  Y - drt_grad
    
    def lr_gen(self, drt_vec, Y):
       min_value = 1e-6
       fn_lip = 0.2
       new_h = self.h - Y @ self.G.T
       new_h[new_h<=min_value] = min_value
       grad = - drt_vec @ self.G.T
       grad[grad<min_value] = min_value
       tmp=torch.div(new_h,grad)
       min_vec,index = torch.min(tmp,dim=1)
       min_vec[min_vec>=fn_lip]=fn_lip
       return  min_vec
    
    @torch.no_grad()
    def mask_mapping(self, m_a, m_b):
        indexarr = torch.arange(m_a.shape[0])
        index_aftera = indexarr[m_a]
        index_afterb = index_aftera[m_b]
        res = torch.zeros(m_a.shape[0], device=self.device)
        res[index_afterb] = 1
        return res
    
    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y
    @torch.no_grad()
    def complete_stepgenY(self,X,Y,step,lr):
        tmp = Y[:,self.partial_vars] 
        newY_partial = step * lr + tmp
        return self.complete_partial(X,newY_partial)


CASE_FNS = dict([(57, case57)])

class ACOPFProblem:
    """
        minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
        s.t.                  p_g min   <= p_g  <= p_g max
                              q_g min   <= q_g  <= q_g max
                              vmag min  <= vmag <= vmag max
                              vang_slack = \theta_slack   # voltage angle     
                              (p_g - p_d) + (q_g - q_d)i = diag(vmag e^{i*vang}) conj(Y) (vmag e^{-i*vang})
    """

    def __init__(self, filename, valid_frac=0.0833, test_frac=0.0833):
        data = spio.loadmat(filename)
        self.nbus = int(filename.split('_')[-1][4:-4])

        ## Define useful power network quantities and indices
        ppc = CASE_FNS[self.nbus]()
        self.ppc = ppc

        self.genbase = ppc['gen'][:, idx_gen.MBASE]
        self.baseMVA = ppc['baseMVA']

        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))

        # indices within gens
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])

        self.ng = ppc['gen'].shape[0]
        self.nslack = len(self.slack)
        self.npv = len(self.pv)

        self.quad_costs = torch.tensor(ppc['gencost'][:,4], dtype=torch.get_default_dtype())
        self.lin_costs  = torch.tensor(ppc['gencost'][:,5], dtype=torch.get_default_dtype())
        self.const_cost = ppc['gencost'][:,6].sum()

        self.pmax = torch.tensor(ppc['gen'][:,idx_gen.PMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.pmin = torch.tensor(ppc['gen'][:,idx_gen.PMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.qmax = torch.tensor(ppc['gen'][:,idx_gen.QMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.qmin = torch.tensor(ppc['gen'][:,idx_gen.QMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.vmax = torch.tensor(ppc['bus'][:,idx_bus.VMAX], dtype=torch.get_default_dtype())
        self.vmin = torch.tensor(ppc['bus'][:,idx_bus.VMIN], dtype=torch.get_default_dtype())
        self.slackva = torch.tensor([np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])], 
            dtype=torch.get_default_dtype()).squeeze(-1)

        ppc2 = deepcopy(ppc)
        ppc2['bus'][:,0] -= 1
        ppc2['branch'][:,[0,1]] -= 1
        Ybus, _, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus.todense()
        self.Ybusr = torch.tensor(np.real(Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(Ybus), dtype=torch.get_default_dtype())

        ## Define optimization problem input and output variables
        demand = data['Dem'].T / self.baseMVA
        gen =  data['Gen'].T / self.genbase
        voltage = data['Vol'].T

        X = np.concatenate([np.real(demand), np.imag(demand)], axis=1)
        Y = np.concatenate([np.real(gen), np.imag(gen), np.abs(voltage), np.angle(voltage)], axis=1)
        feas_mask =  ~np.isnan(Y).any(axis=1)

        self._X = torch.tensor(X[feas_mask], dtype=torch.get_default_dtype())
        self._Y = torch.tensor(Y[feas_mask], dtype=torch.get_default_dtype())
        self._xdim = X.shape[1]
        self._ydim = Y.shape[1]
        self._num = feas_mask.sum()

        self._neq = 2*self.nbus
        self._nineq = 4*self.ng + 2*self.nbus
        self._nknowns = self.nslack

        # indices of useful quantities in full solution
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2*self.ng
        self.va_start_yidx = 2*self.ng + self.nbus


        ## Keep parameters indicating how data was generated
        self.EPS_INTERIOR = data['EPS_INTERIOR'][0][0]
        self.CorrCoeff = data['CorrCoeff'][0][0]
        self.MaxChangeLoad = data['MaxChangeLoad'][0][0]


        ## Define train/valid/test split
        self._valid_frac = valid_frac
        self._test_frac = test_frac


        ## Define variables and indices for "partial completion" neural network

        # pg (non-slack) and |v|_g (including slack)
        self._partial_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv, self.va_start_yidx + self.slack])
        self._other_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)
        self._partial_unknown_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv])

        # initial values for solver
        self.vm_init = ppc['bus'][:, idx_bus.VM]
        self.va_init = np.deg2rad(ppc['bus'][:, idx_bus.VA])
        self.pg_init = ppc['gen'][:, idx_gen.PG] / self.genbase
        self.qg_init = ppc['gen'][:, idx_gen.QG] / self.genbase

        # voltage angle at slack buses (known)
        self.slack_va = self.va_init[self.slack]

        # indices of useful quantities in partial solution
        self.pg_pv_zidx = np.arange(self.npv)
        self.vm_spv_zidx = np.arange(self.npv, 2*self.npv + self.nslack)

        # useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nbus


        ### For Pytorch
        self._device = None


    def __str__(self):
        return 'ACOPF-{}-{}-{}-{}-{}-{}'.format(
            self.nbus,
            self.EPS_INTERIOR, self.CorrCoeff, self.MaxChangeLoad,
            self.valid_frac, self.test_frac)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_unknown_vars

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num * self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num * self.train_frac):int(self.num * (self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num * (self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def get_yvars(self, Y):
        pg = Y[:, :self.ng]
        qg = Y[:, self.ng:2*self.ng]
        vm = Y[:, -2*self.nbus:-self.nbus]
        va = Y[:, -self.nbus:]
        return pg, qg, vm, va

    def obj_fn(self, Y):
        pg, _, _, _ = self.get_yvars(Y)
        pg_mw = pg * torch.tensor(self.genbase).to(self.device)
        cost = (self.quad_costs * pg_mw**2).sum(axis=1) + \
            (self.lin_costs * pg_mw).sum(axis=1) + \
            self.const_cost
        return cost / (self.genbase.mean() ** 2)

    @torch.no_grad()
    def fn_diff_get(self,X,Y):
        pg, _, _, _ = self.get_yvars(Y)
        arg_eff = torch.tensor(self.genbase).to(self.device)
        pg_mw = pg * (arg_eff **2)
        fn_diff = 2*(self.quad_costs * pg_mw)+ self.lin_costs*arg_eff 
        fn_full_diff=torch.zeros(pg.shape[0],self.ydim,device=DEVICE)
        #pg_index_forall=np.concatenate([self.slack, self.pv_])
        #fn_full_diff[:,self.pg_start_yidx+pg_index_forall]=fn_diff
        fn_full_diff[:,self.pg_start_yidx+self.pv_]=fn_diff[:,self.pv_]
        fn_full_diff[:,self.pg_start_yidx+self.slack_]=fn_diff[:,self.slack_]
        eq_jac = self.eq_jac(Y)
        dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]).bmm(eq_jac[:, :, self.partial_vars])
        tmp1 = dynz_dz.transpose(1,2).bmm(fn_full_diff[:,self.other_vars].unsqueeze(-1)).squeeze(-1)
        tmp2 = fn_full_diff[:, self.partial_vars]
        full_partial_diff = tmp1+tmp2

        full_grad = torch.zeros(X.shape[0], self.ydim, device=DEVICE)
        full_grad[:, self.partial_vars] = full_partial_diff
        full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_diff.unsqueeze(-1)).squeeze(-1)
        return full_grad / (self.genbase.mean() ** 2)
    
    @torch.no_grad()
    def get_lip(self):
        gen_double=torch.tensor(self.genbase).to(self.device)**2
        return torch.max(self.quad_costs[1:]*gen_double[1:])  / (self.genbase.mean() ** 2)

    def eq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        ## power balance equations
        tmp1 = vr@self.Ybusr - vi@self.Ybusi
        tmp2 = -vr@self.Ybusi - vi@self.Ybusr

        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=self.device)
        pg_expand[:, self.spv] = pg
        real_resid = (pg_expand - X[:, :self.nbus]) - (vr*tmp1 - vi*tmp2)

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=self.device)
        qg_expand[:, self.spv] = qg
        react_resid = (qg_expand - X[:, self.nbus:]) - (vr*tmp2 + vi*tmp1)

        ## all residuals
        resids = torch.cat([
            real_resid,
            react_resid
        ], dim=1)
        
        return resids

    def ineq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        resids = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            vm - self.vmax,
            self.vmin - vm
        ], dim=1)
        return resids

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        eq_resid = self.eq_resid(X,Y)
        return 2*eq_jac.transpose(1,2).bmm(eq_resid.unsqueeze(-1)).squeeze(-1)

    def ineq_grad(self, X, Y):
        ineq_jac = self.ineq_jac(Y)
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_jac.transpose(1,2).bmm(ineq_dist.unsqueeze(-1)).squeeze(-1)

    """ def ineq_partial_grad(self, X, Y,lr=0):
        eq_jac = self.eq_jac(Y)
        dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]).bmm(eq_jac[:, :, self.partial_vars])

        direct_grad = self.ineq_grad(X, Y)
        indirect_partial_grad = dynz_dz.transpose(1,2).bmm(
            direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

        full_partial_grad = indirect_partial_grad + direct_grad[:, self.partial_vars]

        full_grad = torch.zeros(X.shape[0], self.ydim, device=self.device)
        full_grad[:, self.partial_vars] = full_partial_grad
        full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_grad.unsqueeze(-1)).squeeze(-1)

        return full_grad """
    
    def ineq_partial_grad(self, X, Y, lr):
        eq_jac = self.eq_jac(Y)
        dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]).bmm(eq_jac[:, :, self.partial_vars])

        direct_grad = self.ineq_grad(X, Y)
        indirect_partial_grad = dynz_dz.transpose(1,2).bmm(
            direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

        full_partial_grad = indirect_partial_grad + direct_grad[:, self.partial_vars]

        full_grad = torch.zeros(X.shape[0], self.ydim, device=self.device)
        full_grad[:, self.partial_vars] = full_partial_grad
        full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_grad.unsqueeze(-1)).squeeze(-1)
        full_grad[:, self.other_vars] = torch.inverse(eq_jac[:, :, self.other_vars]).bmm(self.eq_resid(X,Y).unsqueeze(-1)).squeeze(-1) + full_grad[:, self.other_vars]
        return full_grad 
    
    def momentun_fix(self,X,Y,Y_step, momentum, old_Y_step,lr):
        #new_Y_step = lr * Y_step + momentum * old_Y_step
        eq_jac = self.eq_jac(Y)
        dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]).bmm(eq_jac[:, :, self.partial_vars])

        direct_grad = self.ineq_grad(X, Y)
        indirect_partial_grad = dynz_dz.transpose(1,2).bmm(
            direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

        new_Y_step = lr * Y_step + momentum * old_Y_step
        Y_new = Y - new_Y_step

        full_step = torch.zeros(X.shape[0], self.ydim, device=self.device)
        full_step[:, self.partial_vars] = new_Y_step[:, self.partial_vars]
        full_step[:, self.other_vars] =  torch.inverse(eq_jac[:, :, self.other_vars]).bmm(self.eq_resid(X,Y_new).unsqueeze(-1)).squeeze(-1) + new_Y_step[:, self.other_vars]
        return full_step

    def eq_jac(self, Y):
        _, _, vm, va = self.get_yvars(Y)

        # helper functions
        mdiag = lambda v1, v2: torch.diag_embed(v1).bmm(torch.diag_embed(v2))
        Ydiagv = lambda Y, v: Y.unsqueeze(0).expand(v.shape[0], *Y.shape).bmm(torch.diag_embed(v))
        dtm = lambda v, M: torch.diag_embed(v).bmm(M)

        # helper quantities
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        Yr = self.Ybusr
        Yi = self.Ybusi
        YrvrYivi = vr@Yr - vi@Yi
        YivrYrvi = vr@Yi + vi@Yr

        # real power equations
        dreal_dpg = torch.zeros(self.nbus, self.ng, device=self.device) 
        dreal_dpg[self.spv, :] = torch.eye(self.ng, device=self.device)
        dreal_dvm = -mdiag(cosva, YrvrYivi) - dtm(vr, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva)) \
            -mdiag(sinva, YivrYrvi) - dtm(vi, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva))
        dreal_dva = -mdiag(-vi, YrvrYivi) - dtm(vr, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr)) \
            -mdiag(vr, YivrYrvi) - dtm(vi, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr))
        
        # reactive power equations
        dreact_dqg = torch.zeros(self.nbus, self.ng, device=self.device)
        dreact_dqg[self.spv, :] = torch.eye(self.ng, device=self.device)
        dreact_dvm = mdiag(cosva, YivrYrvi) + dtm(vr, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva)) \
            -mdiag(sinva, YrvrYivi) - dtm(vi, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva))
        dreact_dva = mdiag(-vi, YivrYrvi) + dtm(vr, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr)) \
            -mdiag(vr, YrvrYivi) - dtm(vi, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr))

        jac = torch.cat([
            torch.cat([dreal_dpg.unsqueeze(0).expand(vr.shape[0], *dreal_dpg.shape), 
                torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreal_dvm, dreal_dva], dim=2),
            torch.cat([torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreact_dqg.unsqueeze(0).expand(vr.shape[0], *dreact_dqg.shape),
                dreact_dvm, dreact_dva], dim=2)],
            dim=1)

        return jac

    def ineq_jac(self, Y):
        jac = torch.cat([
            torch.cat([torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([-torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device),
                torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device), 
                -torch.eye(self.ng, device=self.device),
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=self.device),
                torch.zeros(self.nbus, self.ng, device=self.device), 
                torch.eye(self.nbus, device=self.device), 
                torch.zeros(self.nbus, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=self.device), 
                torch.zeros(self.nbus, self.ng, device=self.device),
                -torch.eye(self.nbus, device=self.device), 
                torch.zeros(self.nbus, self.nbus, device=self.device)], dim=1)
            ], dim=0)
        return jac.unsqueeze(0).expand(Y.shape[0], *jac.shape)

    # Processes intermediate neural network output
    def process_output(self, X, out):
        out2 = nn.Sigmoid()(out[:, :-self.nbus+self.nslack])
        pg = out2[:, :self.qg_start_yidx] * self.pmax + (1-out2[:, :self.qg_start_yidx]) * self.pmin
        qg = out2[:, self.qg_start_yidx:self.vm_start_yidx] * self.qmax + \
            (1-out2[:, self.qg_start_yidx:self.vm_start_yidx]) * self.qmin
        vm = out2[:, self.vm_start_yidx:] * self.vmax + (1- out2[:, self.vm_start_yidx:]) * self.vmin

        va = torch.zeros(X.shape[0], self.nbus, device=self.device)
        va[:, self.nonslack_idxes] = out[:, self.va_start_yidx:]
        va[:, self.slack] = torch.tensor(self.slack_va, device=self.device).unsqueeze(0).expand(X.shape[0], self.nslack)

        return torch.cat([pg, qg, vm, va], dim=1)

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y_partial = torch.zeros(Z.shape, device=self.device) #Y_partial is part of the network output and will be completed later.

        # Re-scale real powers
        Y_partial[:, self.pg_pv_zidx] = Z[:, self.pg_pv_zidx] * self.pmax[1:] + \
             (1-Z[:, self.pg_pv_zidx]) * self.pmin[1:]
        
        # Re-scale real parts of voltages
        Y_partial[:, self.vm_spv_zidx] = Z[:, self.vm_spv_zidx] * self.vmax[self.spv] + \
            (1-Z[:, self.vm_spv_zidx]) * self.vmin[self.spv]

        return PFFunction(self)(X, Y_partial)

    def wander_step_mod3(self, X, Y,border_shreshold,border_num,active_lr):
        mytestsign = True
        datanum = Y.shape[0]
        zero_shreshold = 1e-7
        ineq_num = self._nineq 
        eq_num = self._neq
        dim_num = self.ydim 
        feasible_num = dim_num - eq_num
        
        random_pick = np.random.choice(feasible_num, feasible_num, replace=False)
        random_pos =self.partial_vars[random_pick]
        
        eq_jac = self.eq_jac(Y)
        dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]).bmm(eq_jac[:, :, self.partial_vars])
        
        ineq_diff=self.ineq_jac(Y)
        G_effective = ineq_diff[:,:,self.partial_vars] + ineq_diff[:,:, self.other_vars].bmm(dynz_dz)
        border_value = self.ineq_resid(X, Y)  

        border_sortedvalue, bordersorted_index = torch.sort(border_value,descending = True,dim = -1)
        border_index = bordersorted_index[:,:border_num]
        border_index[border_sortedvalue[:,:border_num]<border_shreshold] =  ineq_num
       
        fn_grad = self.fn_diff_get(X,Y)[:,random_pick]
       
        ineq_tmp= torch.zeros((datanum,ineq_num + 1, feasible_num), device=self.device)
        ineq_tmp[:,0:-1, :] = G_effective[:,:,random_pick]
        ineq_tmp[:,-1, :] = 0
        data_num_index =  torch.range(0,datanum-1,dtype=torch.long,device=DEVICE)
        ineq_matrix = ineq_tmp[data_num_index.view(-1,1), border_index,:]
        
        satisified_sign = torch.ones(datanum,dtype=torch.int32,device=self.device) 
        satisified_num = torch.zeros(datanum,dtype=torch.int32,device=self.device) 
        zero_sign = torch.zeros(datanum,dtype=torch.int32,device=self.device) 
        
        for iter in range(border_num):
            
            if ~torch.any(satisified_sign.bool()):
                break
            sub_ineq = ineq_matrix[satisified_sign.bool(),:,:]
            sub_fn = fn_grad[satisified_sign.bool(),:]
            
            diff_minval, diff_minindex = torch.min(torch.mul(sub_fn[:,iter:].view(-1,1,feasible_num-iter),sub_ineq[:,iter:,iter:]).sum(dim=2),dim=1)
            subsatisify_sign = diff_minval >= 0
            subsatisify_inv = subsatisify_sign == False
            
            divzerosign_bool = torch.abs(sub_ineq[:,iter,iter]) <= zero_shreshold
            divnozerosign_bool = (divzerosign_bool==False)
            
            sub_ineq[divnozerosign_bool,iter+1:,iter+1:] = sub_ineq[divnozerosign_bool,iter+1:,iter+1:] - \
                torch.mul(sub_ineq[divnozerosign_bool,iter:iter+1,iter+1:],torch.div(sub_ineq[divnozerosign_bool,iter+1:,iter:iter+1],sub_ineq[divnozerosign_bool,iter:iter+1,iter:iter+1]))
            
            sub_fn[divnozerosign_bool,iter+1:] = sub_fn[divnozerosign_bool,iter+1:] - \
                torch.mul(sub_ineq[divnozerosign_bool,iter,iter+1:],torch.div(sub_fn[divnozerosign_bool,iter:iter+1],sub_ineq[divnozerosign_bool,iter:iter+1,iter]))
           
            test_sign = (divzerosign_bool & subsatisify_inv) 
            if test_sign.numel():
                zero_sign[satisified_sign.bool()] = test_sign.int()
            
            satisified_sign[satisified_sign.bool()] = subsatisify_inv.int() 
            
            ineq_matrix[satisified_sign.bool(),:,:] = sub_ineq[subsatisify_inv,:,:] 
            fn_grad[satisified_sign.bool(),:] = sub_fn[subsatisify_inv,:]
            
            signofzero = satisified_sign.bool() & zero_sign.bool() 
            if torch.any(signofzero):
                satisified_sign[signofzero] = 0 
            satisified_num = satisified_num + satisified_sign
        
        step = 0.05
        for iter in range(border_num):
            deal_sign = satisified_num >= (border_num-iter)
            if torch.any(deal_sign):
                fn_grad[deal_sign,border_num-iter-1] = torch.div(-torch.mul(fn_grad[deal_sign,border_num-iter:], \
                    ineq_matrix[deal_sign, border_num-iter-1, border_num-iter:]).sum(dim=1),ineq_matrix[deal_sign, border_num-iter-1, border_num-iter-1])
       
        #还原到100维度
        #drt_grad = torch.zeros(datanum,dim_num,device=self.device)
        #drt_grad[:, random_pos] = fn_grad
        #drt_grad[:, self.other_vars] = - ( drt_grad[:, self.partial_vars] @ self._A_partial.T) @ self._A_other_inv.T
        #lr_rate生成
        min_value = -1e-6
        div_minvalue = 1e-8
        border_value[border_value>=min_value] = min_value
        
       
        indeedgrad = torch.zeros_like(fn_grad)
        indeedgrad[:,random_pick] = fn_grad
    
        grad_multi = -indeedgrad.view(datanum,1,-1).bmm(G_effective.transpose(1,2)).squeeze(1)
        
        grad_multi[grad_multi<div_minvalue] = div_minvalue
        tmp=torch.div(-border_value,grad_multi)
        
        min_vec,index = torch.min(tmp,dim=1)
    
        min_vec[ min_vec>=active_lr] = active_lr 
        tmp_fn = torch.mul(indeedgrad, min_vec.view(-1,1))
            
        drt_grad = torch.zeros(datanum,dim_num,device=self.device)    
        drt_grad[:, self.partial_vars] = tmp_fn
        drt_grad[:, self.other_vars] =   tmp_fn.view(datanum,1,-1).bmm(dynz_dz.transpose(1,2)).squeeze(1)
        
        Y_new = Y - drt_grad*0.05
        drt_grad[:, self.other_vars] =  20*torch.inverse(eq_jac[:, :, self.other_vars]).bmm(self.eq_resid(X,Y_new).unsqueeze(-1)).squeeze(-1) + drt_grad[:, self.other_vars]
        Y_new = Y - drt_grad*0.05
        #other attemption
        """ for i_d in range(datanum):
            tmp_border=(self.ineq_resid(X, Y)[i_d,:].view(-1,1).repeat(1,dim_num))>0
            repair_ineq_jac= self.ineq_jac(Y)[i_d,:,:][tmp_border].view(-1,dim_num)
            select_ineq_num=repair_ineq_jac.shape[0]
            if 0 != select_ineq_num:
                try:
                    drt_grad[i_d, :select_ineq_num] = 20*torch.inverse(repair_ineq_jac[:,:select_ineq_num]).bmm( \
                        self.ineq_resid(X, Y)[i_d,:,:].unsqueeze(-1)).squeeze(-1) + drt_grad[i_d, self.other_vars]
                except torch._C._LinAlgError:
                    drt_grad=0 """
    
        """ for iter in range(border_num):
            deal_sign = satisified_num >= (border_num-iter)
            ineq_error_fixed_step =  111
            
            if torch.any(deal_sign):
                Y_new[deal_sign,border_num-iter-1] = Y_new - Y_new """


        return  Y - drt_grad*0.05





    def opt_solve(self, X, solver_type='pypower', tol=1e-4):
        X_np = X.detach().cpu().numpy()

        ppc = self.ppc

        # Set reduced voltage bounds if applicable
        ppc['bus'][:,idx_bus.VMIN] = ppc['bus'][:,idx_bus.VMIN] + self.EPS_INTERIOR
        ppc['bus'][:,idx_bus.VMAX] = ppc['bus'][:,idx_bus.VMAX] - self.EPS_INTERIOR

        # Solver options
        ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol)  # MIPS PDIPM

        Y = []
        total_time = 0
        for i in range(X_np.shape[0]):
            print(i)
            ppc['bus'][:, idx_bus.PD] = X_np[i, :self.nbus] * self.baseMVA
            ppc['bus'][:, idx_bus.QD] = X_np[i, self.nbus:] * self.baseMVA

            start_time = time.time()
            my_result = opf(ppc, ppopt)
            end_time = time.time()
            total_time += (end_time - start_time)

            pg = my_result['gen'][:, idx_gen.PG] / self.genbase
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))

        return np.array(Y), total_time, total_time/len(X_np)


class nonconvex_ipopt(object):
    def __init__(self, Q, p, A, G):
        self.Q = Q
        self.p = p
        self.A = A
        self.G = G
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p@np.sin(y)

    def gradient(self, y):
        return self.Q@y + (self.p * np.cos(y))

    def constraints(self, y):
        return np.hstack([self.A@y, self.G@y])

    def jacobian(self, y):
        return np.concatenate([self.A.flatten(), self.G.flatten()])

    # # Don't use: In general, more efficient with numerical approx
    # def hessian(self, y, lagrange, obj_factor):
    #     H = obj_factor * (self.Q - np.diag(self.p * np.sin(y)) )
    #     return H[self.tril_indices]

    # def intermediate(self, alg_mod, iter_count, obj_value,
    #         inf_pr, inf_du, mu, d_norm, regularization_size,
    #         alpha_du, alpha_pr, ls_trials):
    #     print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


def PFFunction(data, tol=1e-5, bsz=200, max_iters=50):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):

            ## Step 1: Newton's method
            Y = torch.zeros(X.shape[0], data.ydim, device=DEVICE)
            
            # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
            Y[:, data.pg_start_yidx + data.pv_] = Z[:, data.pg_pv_zidx]    # pg at non-slack gens
            Y[:, data.vm_start_yidx + data.spv] = Z[:, data.vm_spv_zidx]   # vm at gens
            Y[:, data.va_start_yidx + data.slack] = torch.tensor(data.slack_va, device=DEVICE)  # va at slack bus

            # init guesses for remaining values，吧剩余的y补全
            Y[:, data.vm_start_yidx + data.pq] = torch.tensor(data.vm_init[data.pq], device=DEVICE)  # vm at load buses
            Y[:, data.va_start_yidx + data.pv] = torch.tensor(data.va_init[data.pv], device=DEVICE)  # va at non-slack gens 
            Y[:, data.va_start_yidx + data.pq] = torch.tensor(data.va_init[data.pq], device=DEVICE)  # va at load buses
            Y[:, data.qg_start_yidx:data.qg_start_yidx+data.ng] = 0    # qg at gens (not used in Newton upd)
            Y[:, data.pg_start_yidx+data.slack_] = 0                   # pg at slack (not used in Newton upd)

            keep_constr = np.concatenate([
                data.pflow_start_eqidx + data.pv,     # real power flow at non-slack gens
                data.pflow_start_eqidx + data.pq,     # real power flow at load buses
                data.qflow_start_eqidx + data.pq])    # reactive power flow at load buses
            newton_guess_inds = np.concatenate([             
                data.vm_start_yidx + data.pq,         # vm at load buses
                data.va_start_yidx + data.pv,         # va at non-slack gens
                data.va_start_yidx + data.pq])        # va at load buses

            converged = torch.zeros(X.shape[0])
            jacs = []
            newton_jacs_inv = []
            for b in range(0, X.shape[0], bsz):
                # print('batch: {}'.format(b))
                X_b = X[b:b+bsz]
                Y_b = Y[b:b+bsz]

                for i in range(max_iters):
                    # print(i)
                    gy = data.eq_resid(X_b, Y_b)[:, keep_constr]
                    jac_full = data.eq_jac(Y_b)
                    jac = jac_full[:, keep_constr, :]  #只有一部分等式completion
                    newton_jac_inv = torch.inverse(jac[:, :, newton_guess_inds])
                    delta = newton_jac_inv.bmm(gy.unsqueeze(-1)).squeeze(-1)
                    Y_b[:, newton_guess_inds] -= delta
                    if torch.norm(delta, dim=1).abs().max() < tol:
                        break

                converged[b:b+bsz] = (delta.abs() < tol).all(dim=1)
                jacs.append(jac_full)
                newton_jacs_inv.append(newton_jac_inv)
            #Y[b:b+bsz] = Y_b    #似乎少了这么一段

            ## Step 2: Solve for remaining variables

            # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.qflow_start_eqidx + data.spv]
            # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
            Y[:, data.pg_start_yidx + data.slack_] = \
                -data.eq_resid(X, Y)[:, data.pflow_start_eqidx + data.slack]

            ctx.data = data
            ctx.save_for_backward(torch.cat(jacs), torch.cat(newton_jacs_inv),
                torch.tensor(newton_guess_inds, device=DEVICE), 
                torch.tensor(keep_constr, device=DEVICE))

            return Y

        @staticmethod
        def backward(ctx, dl_dy):

            data = ctx.data
            jac, newton_jac_inv, newton_guess_inds, keep_constr = ctx.saved_tensors

            ## Step 2 (calc pg at slack and qg at gens)

            # gradient of all voltages through step 3 outputs
            last_eqs = np.concatenate([data.pflow_start_eqidx + data.slack, data.qflow_start_eqidx + data.spv])
            last_vars = np.concatenate([
                data.pg_start_yidx + data.slack_, np.arange(data.qg_start_yidx, data.qg_start_yidx + data.ng)])
            jac3 = jac[:, last_eqs, :]
            dl_dvmva_3 = -jac3[:, :, data.vm_start_yidx:].transpose(1,2).bmm(
                dl_dy[:, last_vars].unsqueeze(-1)).squeeze(-1)

            # gradient of pd at slack and qd at gens through step 3 outputs
            dl_dpdqd_3 = dl_dy[:, last_vars]

            # insert into correct places in x and y loss vectors
            dl_dy_3 = torch.zeros(dl_dy.shape, device=DEVICE)
            dl_dy_3[:, data.vm_start_yidx:] = dl_dvmva_3

            dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=DEVICE)
            dl_dx_3[:, np.concatenate([data.slack, data.nbus + data.spv])] = dl_dpdqd_3


            ## Step 1
            dl_dy_total = dl_dy_3 + dl_dy  # Backward pass vector including result of last step

            # Use precomputed inverse jacobian
            jac2 = jac[:, keep_constr, :]
            d_int = newton_jac_inv.transpose(1,2).bmm(
                            dl_dy_total[:,newton_guess_inds].unsqueeze(-1)).squeeze(-1)

            dl_dz_2 = torch.zeros(dl_dy.shape[0], data.npv + data.ng, device=DEVICE)
            dl_dz_2[:, data.pg_pv_zidx] = -d_int[:, :data.npv]  # dl_dpg at pv buses
            dl_dz_2[:, data.vm_spv_zidx] = -jac2[:, :, data.vm_start_yidx + data.spv].transpose(1,2).bmm(
                d_int.unsqueeze(-1)).squeeze(-1)

            dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=DEVICE)
            dl_dx_2[:, data.pv] = d_int[:, :data.npv]                       # dl_dpd at pv buses
            dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv+len(data.pq)]  # dl_dpd at pq buses
            dl_dx_2[:, data.nbus + data.pq] = d_int[:, -len(data.pq):]      # dl_dqd at pq buses


            # Final quantities
            dl_dx_total = dl_dx_3 + dl_dx_2
            dl_dz_total = dl_dz_2 + dl_dy_total[:, np.concatenate([
                data.pg_start_yidx + data.pv_, data.vm_start_yidx + data.spv])]

            return dl_dx_total, dl_dz_total


    return PFFunctionFn.apply

class HighorderProblem(object):
    """ 
        minimize_y 1/2 * y^(2T) Q y^2 + p^Ty
        s.t.       Ay =  x
                   Gy <= h
    """
    def __init__(self, Q, p, A, G, h, X, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._G = torch.tensor(G)
        self._h = torch.tensor(h)
        self._X = torch.tensor(X)
        self._Y = None
        self._xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = G.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        det = 0
        i = 0
        while abs(det) < 0.000001 and i < 100: # abs(det) < 0.0001 and i < 100:
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
            det = torch.det(self._A[:, self._other_vars]) #Prevent linear dependence.
            i += 1
        if i == 100:
            raise Exception
        else:
            self._A_partial = self._A[:, self._partial_vars]
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])

        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'SimpleProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def h(self):
        return self._h

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num*self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def obj_fn(self, Y):
        Ydb= torch.mul(Y,Y)
        return (0.5*(Ydb@self.Q)*Ydb + self.p*Y).sum(dim=1)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_dist@self.G
    
    #there are no lip for such problem, hence we set it as the hyperparameter
    @torch.no_grad()
    def get_lip(self):
        return 0.001
    
    #(deprecated)here are no lip for such problem, hence we use 1/2 second order gradient to approximate it 
    def get_lip_cal(self,Y):
        grad = 0.5*4*self.Q @ torch.mul(torch.mul(Y,Y),Y)
        second_grad = 0.5*12 * self.Q @ torch.mul(Y,Y)
        afterdealeq_diff = second_grad[:,self.partial_vars] - second_grad[:,self.other_vars]  @  (self._A_other_inv @ self._A_partial)
        return 0.5* torch.max(afterdealeq_diff, dim=0)
        
    #obtain the gradient after correction 
    def ineq_partial_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) #the new coefficient G after completion 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  #the new coefficient h after completion 
        grad = 2 * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective 
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y
    
    def newton_corr_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T 
        G_effectivesum = torch.norm(G_effective,dim=1,p=2)
        grad = torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ (torch.div(G_effective,G_effectivesum.view(-1,1))/2500)
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] =  grad
        Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y
    
    @torch.no_grad()
    def fn_diff_get(self,Y):
        fn_grad = 0.5*4*torch.mul(torch.mul(Y,Y),Y) @ self.Q  
        afterdealeq_diff = fn_grad[:,self.partial_vars] - fn_grad[:,self.other_vars]  @  (self._A_other_inv @ self._A_partial)
        #cmp_fndiff = torch.zeros(Y.shape[0], self.ydim, device=self.device)
        #cmp_fndiff[:, self.partial_vars] =  afterdealeq_diff
        #cmp_fndiff[:, self.other_vars] = - (afterdealeq_diff @ self._A_partial.T) @ self._A_other_inv.T
        #return cmp_fndiff
        return afterdealeq_diff
    
    @torch.no_grad()   
    def ineq_diff_get(self):
        return self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
    
    
    @torch.no_grad() 
    def getborderindex(self,X,Y,border_shreshold,border_num):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  
        border_value = Y[:, self.partial_vars] @ G_effective.T - h_effective 
        border_sortedvalue, bordersorted_index = torch.sort(border_value,descending = True,dim = -1)
        NAN_value_num = border_value.shape[1] 
        final_index=torch.zeros((border_value.shape[0],border_num),dtype=torch.int64,device=DEVICE)
        for i in range(border_value.shape[0]):
            tmp_index= bordersorted_index[i,:][border_sortedvalue[i,:]>=border_shreshold]
            inequality_index = torch.ones(border_value.shape[1],dtype=torch.int64,device=DEVICE)*NAN_value_num
            inequality_index[0:min(border_num,tmp_index.shape[0])] = tmp_index[0:min(border_num,tmp_index.shape[0])]
            final_index[i,:] = inequality_index[0:border_num]
        return final_index
              
    @torch.no_grad()
    def lr_gen1(self, drt_vec, Y, border_index,satisified_num):
       min_value = 1e-6
       fn_lip = 0.2
       new_h = self.h - Y @ self.G.T
       #print("befor:",new_h[0:10,3])
       new_h[new_h<=min_value] = min_value
       #print("after:",new_h[0:10,3])
       grad = - drt_vec @ self.G.T
       grad[grad<min_value] = min_value
       tmp=torch.div(new_h,grad)
       for i in range(border_index.shape[0]):
           kk_tmp=border_index[i,0:int(satisified_num[i])]
           bool_vec = [kk_tmp<self._nineq]
           tmp_index=border_index[i,0:int(satisified_num[i])][bool_vec]
           tmp[i,tmp_index]= 0.2
       #print(grad[0:10,3])
       min_vec,index = torch.min(tmp,dim=1)
       min_vec[min_vec>=fn_lip]=fn_lip
       return  min_vec
    
    def lr_gen(self, drt_vec, Y):
       min_value = 1e-6
       fn_lip = 0.2
       new_h = self.h - Y @ self.G.T
       new_h[new_h<=min_value] = min_value
       grad = - drt_vec @ self.G.T
       grad[grad<min_value] = min_value
       tmp=torch.div(new_h,grad)
       min_vec,index = torch.min(tmp,dim=1)
       min_vec[min_vec>=fn_lip]=fn_lip
       return  min_vec
        
    def wander_step_mod3(self, X, Y,border_shreshold,border_num,active_lr):
        mytestsign = True
        datanum = Y.shape[0]
        zero_shreshold = 1e-7
        ineq_num = self._nineq #number of inequation
        eq_num = self._neq #number of equation
        dim_num = self.ydim #dimension of y
        feasible_num = dim_num - eq_num
        #position swap for random selection(simulate random)
        random_pick = np.random.choice(feasible_num, feasible_num, replace=False)
        random_pos =self.partial_vars[random_pick]
        #generate the boundary condition
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  
        border_value = Y[:, self.partial_vars] @ G_effective.T - h_effective 
        border_sortedvalue, bordersorted_index = torch.sort(border_value,descending = True,dim = -1)
        border_index = bordersorted_index[:,:border_num]
        border_index[border_sortedvalue[:,:border_num]<border_shreshold] =  ineq_num
        #innitialize the gradient
        fn_grad = self.fn_diff_get(Y)[:,random_pick]
        #construct the boundary inequality matrix
        ineq_tmp= torch.zeros((ineq_num + 1, feasible_num), device=self.device)
        ineq_tmp[0:-1, :] = G_effective[:,random_pick]
        ineq_tmp[-1, :] = 0
        ineq_matrix = ineq_tmp[ border_index,:]
        #initialize the signal
        satisified_sign = torch.ones(datanum,dtype=torch.int32,device=self.device) #determine whether the algorithm has terminated: 1 for "cannot terminate" and 0 for "can terminate."
        satisified_num = torch.zeros(datanum,dtype=torch.int32,device=self.device) #record in which iteration BWS finished
        zero_sign = torch.zeros(datanum,dtype=torch.int32,device=self.device) #Check if 0 is detected; if detected, skip it.
        for iter in range(border_num):
            #form equivalent problems through slicing
            if ~torch.any(satisified_sign.bool()):
                break
            sub_ineq = ineq_matrix[satisified_sign.bool(),:,:]
            sub_fn = fn_grad[satisified_sign.bool(),:]
            #determine if the conditions are satisfied
            diff_minval, diff_minindex = torch.min(torch.mul(sub_fn[:,iter:].view(-1,1,feasible_num-iter),sub_ineq[:,iter:,iter:]).sum(dim=2),dim=1)
            subsatisify_sign = diff_minval >= 0
            subsatisify_inv = subsatisify_sign == False
            #Update the inequalities
            #Prevent linear dependence among inequalities matrix
            divzerosign_bool = torch.abs(sub_ineq[:,iter,iter]) <= zero_shreshold
            divnozerosign_bool = (divzerosign_bool==False)
            if ~torch.any(divnozerosign_bool):
                fn_grad[satisified_sign.bool(),:]=0
                break 
            #handle inequalities that do not start close to 0.
            sub_ineq[divnozerosign_bool,iter+1:,iter+1:] = sub_ineq[divnozerosign_bool,iter+1:,iter+1:] - \
                torch.mul(sub_ineq[divnozerosign_bool,iter:iter+1,iter+1:],torch.div(sub_ineq[divnozerosign_bool,iter+1:,iter:iter+1],sub_ineq[divnozerosign_bool,iter:iter+1,iter:iter+1]))
            #handle function gradients that do not start close to 0.
            sub_fn[divnozerosign_bool,iter+1:] = sub_fn[divnozerosign_bool,iter+1:] - \
                torch.mul(sub_ineq[divnozerosign_bool,iter,iter+1:],torch.div(sub_fn[divnozerosign_bool,iter:iter+1],sub_ineq[divnozerosign_bool,iter:iter+1,iter]))

            #signal update
            test_sign = (divzerosign_bool & subsatisify_inv) 
            if test_sign.numel():
                zero_sign[satisified_sign.bool()] = test_sign.int()
            #Update the global state based on the sign
            satisified_sign[satisified_sign.bool()] = subsatisify_inv.int() 
            #update the global function and inequalities
            ineq_matrix[satisified_sign.bool(),:,:] = sub_ineq[subsatisify_inv,:,:] 
            fn_grad[satisified_sign.bool(),:] = sub_fn[subsatisify_inv,:]
            #update the global state 
            signofzero = satisified_sign.bool() & zero_sign.bool() 
            if torch.any(signofzero):
                satisified_sign[signofzero] = 0 
            satisified_num = satisified_num + satisified_sign
        #recover process
        for iter in range(border_num):
            deal_sign = satisified_num >= (border_num-iter)
            if torch.any(deal_sign):
                fn_grad[deal_sign,border_num-iter-1] = torch.div(-torch.mul(fn_grad[deal_sign,border_num-iter:], \
                    ineq_matrix[deal_sign, border_num-iter-1, border_num-iter:]).sum(dim=1),ineq_matrix[deal_sign, border_num-iter-1, border_num-iter-1])
        #drt_grad = torch.zeros(datanum,dim_num,device=self.device)
        #drt_grad[:, random_pos] = fn_grad
        #drt_grad[:, self.other_vars] = - ( drt_grad[:, self.partial_vars] @ self._A_partial.T) @ self._A_other_inv.T
        #generate the lr_rate 
        min_value = -1e-6
        div_minvalue = 1e-8
        border_value[border_value>=min_value] = min_value
        
       
        indeedgrad = torch.zeros_like(fn_grad)
        indeedgrad[:,random_pick] = fn_grad
    
        grad_multi = -indeedgrad @ G_effective.T
        #prevent division by zero.
        grad_multi[grad_multi<div_minvalue] = div_minvalue
        tmp=torch.div(-border_value,grad_multi)
        #print(grad[0:10,3])
        min_vec,index = torch.min(tmp,dim=1)
    
        min_vec[ min_vec>=active_lr] = active_lr 
        tmp_fn = torch.mul(indeedgrad, min_vec.view(-1,1))
            
        drt_grad = torch.zeros(datanum,dim_num,device=self.device)    
        drt_grad[:, self.partial_vars] = tmp_fn
        drt_grad[:, self.other_vars] = - ( tmp_fn @ self._A_partial.T) @ self._A_other_inv.T
        return  Y - drt_grad
        
    @torch.no_grad()
    def ineq_partial_grad_nograd(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial) 
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T  
        grad = 2 * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective 
        Y_step = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y_step[:, self.partial_vars] = grad
        Y_step[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y_step
    
    #d(fn)/d(y'), here, y' represents the free variables after completion
    @torch.no_grad()
    def fn_partial_grad_nograd(self, X, Y):
        fn_diff_Ycom = self.p + 0.5 * Y @ (self.Q.T+self.Q)
        fn_diff_Y = fn_diff_Ycom[:,self.partial_vars] + fn_diff_Ycom[:,self.other_vars] @ (self._A_other_inv @ self._A_partial)
        fn_grad = torch.zeros(X.shape[0], self.ydim, device=self.device)
        fn_grad[:, self.partial_vars] = fn_diff_Y
        fn_grad[:, self.other_vars] = - (fn_diff_Y @ self._A_partial.T) @ self._A_other_inv.T
        return fn_grad

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y
    
    @torch.no_grad()
    def complete_stepgenY(self,X,Y,step,lr):
        tmp = Y[:,self.partial_vars] 
        newY_partial = step * lr + tmp
        return self.complete_partial(X,newY_partial)
    
    
