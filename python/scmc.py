'''
Created on Nov 14, 2016

@author: daqingy
'''

import numpy as np
from scipy.stats import norm, uniform
from scipy import optimize
from abc import abstractmethod

class RandomVariable(object):
    
    def __init__(self, dim):
        self.dim = dim
    
    @abstractmethod
    def sample(self, N):
        sample = np.zeros(N,self.dim)
        return sample
    
    @abstractmethod
    def logpdf(self, x):
        return 0.0    
    
class UniformRandomVariable(RandomVariable):    
    
    def __init__(self, dim, srng):
        self.dim = dim
        self.srng = srng
        self.RVS = []
        for i in range(self.dim):
            X = uniform( loc=srng[i][0], scale=srng[i][1]-srng[i][0] )
            self.RVS.append(X)    
        
    def sample(self, N):
        ins = np.zeros((N,self.dim))
        for i in range(self.dim):
            ins[:,i] = self.RVS[i].rvs(N)
        return ins
           
    def logpdf(self, x):
        lp = 0.0
        for i in range(self.dim):
            lp += self.RVS[i].logpdf(x[i])
        return lp       

class NormalRandomVariable(RandomVariable):    
    
    def __init__(self, dim, mu, var):
        self.dim = dim
        self.RVS = []
        for i in range(self.dim):
            X = norm( loc=mu[i], scale=var[i] )
            self.RVS.append(X)    
        
    def sample(self, N):
        ins = np.zeros((N,self.dim))
        for i in range(self.dim):
            ins[:,i] = self.RVS[i].rvs(N)
        return ins
           
    def logpdf(self, x):
        lp = 0.0
        for i in range(self.dim):
            lp += self.RVS[i].logpdf(x[i])
        return lp       

# log-posterior (equivalent to the constraint if a uniform sample is drawn)
def log_posterior(sample, tau_t, constraint_func, RV_X):
    
    term = constraint_func(sample)
    return np.sum( norm.logcdf(- term * tau_t) ) + RV_X.logpdf(sample)

def Metroplis(x, t_var, tau_t, lpden, log_constraint_func, RV_X):
    
    dim = len(x)
    for d in range(dim):    
        # use a Gaussian jumping distribution
        delta = np.random.normal(loc=0, scale=t_var[d], size=1)
        newx = np.copy(x)
        newx[d] = newx[d] + delta
        lpnum = log_constraint_func(newx, tau_t, RV_X)
        # acceptance ratio
        ratio = lpnum - lpden
        prob = min(1., np.exp(ratio))
        u = np.random.rand()
        if u <= prob:
            x = newx
            lpden = lpnum

    return x, lpden

def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j-1)
    return indices

# adaptive specification of the constraint parameter
def adapt_seq(tau, tau_prev, N, sample, W, log_constraint_func, RV_X):
    
    omega = np.zeros(N)
    for i in range(N):
        cons1 = log_constraint_func(sample[i,:], tau, RV_X)
        cons2 = log_constraint_func(sample[i,:], tau_prev, RV_X)
        omega[i] = cons1 - cons2
  
    W = W * np.exp(omega)
    W = W / np.sum(W)
    
    ESS = 0.0
    if np.sum(np.isnan(W)) != N:
        ESS = 1/np.sum(W**2)   
    return ESS - (N / 2)

def scmc(RV_X, N, M, log_constraint_func, tau_T):
    
    dim = RV_X.dim
    
    sample_seq = []
    lpden_seq = []
    W_seq = []
    tau_seq = [1e-20]
    ESS = [0]
    
    sample0 = RV_X.sample(N)
    sample_seq.append(sample0)
    lpden = np.zeros(N)
    for i in range(N):
        lpden[i] = log_constraint_func(sample0[i,:], tau_seq[0], RV_X)
    lpden_seq.append(lpden)
    W0 = np.ones(N)*(1.0/N)
    W_seq.append(W0)
    
    # sampling at each time step
    t = 0
    while True:
        t = t+1
        
        print "@" + str(t) + " : " + str(tau_seq[t-1])
        
        newsample = sample_seq[t-1]
        newlpden = lpden_seq[t-1]
        newW = W_seq[t-1]
        
        # generate new tau_t
        if adapt_seq(tau_T, tau_seq[t-1], N, newsample, newW, log_constraint_func, RV_X) > 0:
            # close enough, move to tau_T
            print "last step"
            tau_seq.append(tau_T) 
            
        else:
            # the effective sample size is less than N/2 
            # need to get a new tau_t that moves ESS to N/2
            lb = 1e-5
            ub = tau_T 
            if t!= 1:
                lb = tau_seq[t-1]    
                             
            result = optimize.brenth(adapt_seq, lb, ub, args= (tau_seq[t-1], N, newsample, newW, log_constraint_func, RV_X))
            
            #print adapt_seq(result, tau_seq[t-1], N, newsample, newW, constraint_func)
            
            tau_seq.append(result)
                
        omega = np.zeros(N)
        for i in range(N):
            constraint1 = log_constraint_func(newsample[i], tau_seq[t], RV_X)
            constraint2 = log_constraint_func(newsample[i], tau_seq[t-1], RV_X)
            omega[i] = constraint1 - constraint2 
        newW = newW * np.exp(omega)
        # normalize
        newW = newW / np.sum(newW)
        newESS = 1 / sum(newW**2)
        
        ESS.append( newESS )
                
        # resample by weight
        index = np.random.choice(np.arange(0,N,1), size=N, p=newW, replace=True )
        #index = resample(newW)
        newsample = newsample[index,:]
        newlpden = newlpden[index]
        newW = np.ones(N)*(1.0/N)
        
        t_var = np.std(newsample, axis=0) / (t+1)
        
        # sample from K^t
        for j in range(M):
            for i in range(N):
                newsample[i,:], newlpden[i] = Metroplis(newsample[i,:], t_var, tau_seq[t], newlpden[i], log_constraint_func, RV_X)   
        
        sample_seq.append( newsample )
        lpden_seq.append( newlpden )
        W_seq.append( newW ) 
        
        if tau_seq[t] >= tau_T:
            break
    
    sample = sample_seq[t]
    W = W_seq[t]
    lpden = lpden_seq[t]
    return sample, W, lpden