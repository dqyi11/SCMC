'''
Created on Nov 14, 2016

@author: daqingy
'''

import numpy as np
from scipy.stats import norm
from scipy import optimize


def unrestricted_sampling(N, dim, srng):
    
    X = np.random.rand(N, dim)
    for i in range(dim):
        X[:,i] = srng[i][0] + (srng[i][1]-srng[i][0])*X[:,i]
    return X

# log-posterior (equivalent to the constraint if a uniform sample is drawn)
def log_posterior(sample, tau_t, constraint_func):
    
    term = constraint_func(sample)
    return np.sum( norm.logcdf(- term / tau_t) )

def Gibbs(x, q, d, a, tau_t, lpden, constraint_func):
    
    delta = np.random.normal(loc=0, scale=q[d], size=1)
    newx = x
    newx[d] = newx[d] + delta
    lpnum = log_posterior(newx, tau_t, constraint_func)
    ratio = lpnum - lpden
    prob = min(1., np.exp(ratio))
    u = np.random.rand()
    if u <= prob:
        x = newx
        lpden = lpnum
        a = a + 1

    return x, a, lpden

# adaptive specification of the constraint parameter
def adapt_seq(tau, tau_prev, t, N, sample, Wt, constraint_func):
    
    wt = np.zeros(N)
    for i in range(N):
        term = constraint_func(sample[i])
        cons1 = np.sum( norm.logcdf(- term / tau) )
        cons2 = np.sum( norm.logcdf(- term / tau_prev) )
        wt[i] = cons1 - cons2
  
    Wt = Wt * np.exp(wt)
    Wt = Wt / np.sum(Wt)
    
    ESS = 0.0
    if np.sum(np.isnan(Wt)) != N:
        ESS = 1/np.sum(Wt**2)
    return ESS - (N / 2)

def scmc(N, dim, M, L, srng, constraint_func, tau_T= 1e-3, qt = 1):
    
    sample_seq = []
    lpden_seq = []
    W_seq = []
    tau_seq = [1e8]
    b = np.arange(1.5,.1,(.1-1.5)/L)
    a = np.zeros((L,dim))
    tau_seq_0 = np.hstack(([np.inf], b**7))
    ESS = []
    
    # initial sampling on the hypercube
    sample0 = unrestricted_sampling(N, dim, srng)
    sample_seq.append(sample0)
    lpden = np.zeros(N)
    for i in range(N):
        lpden[i] = log_posterior(sample0[i], tau_seq[0], constraint_func)
    lpden_seq.append(lpden)
    W_seq.append(np.ones(N)*(1.0/N))
    
    # sampling at each time step
    t = 0
    while True:
        t = t+1
        
        print "@" + str(t) + " : " + str(tau_seq[t-1])
        
        newsample = sample_seq[t-1]
        newlpden = lpden_seq[t-1]
        newW = W_seq[t-1]
        
        # generate new tau_t
        if adapt_seq(tau_T, tau_seq[t-1], t, N, newsample, newW, constraint_func) > 0:
            tau_seq.append(tau_T) 
        else:
            # the effective sample size is less than N/2 
            # need to get a new tau_t that moves ESS to N/2
            lb = tau_T
            ub = 1e6
            if t < 2:
                lb = min(.1, tau_seq_0[t])            
            if t!= 1:
                ub = tau_seq[t-1]    
                           
            tmp_var1 = adapt_seq(lb, tau_seq[t-1], t, N, newsample, newW, constraint_func)    
            tmp_var2 = adapt_seq(ub, tau_seq[t-1], t, N, newsample, newW, constraint_func)   
            result = optimize.brenth(adapt_seq, lb, ub, args= (tau_seq[t-1], t, N, newsample, newW, constraint_func))
            tau_seq.append(result)
                
        wt = np.zeros(N)
        for i in range(N):
            term = constraint_func(newsample[i])
            constraint1 = np.sum( norm.logcdf( - term / tau_seq[t] )  )
            constraint2 = np.sum( norm.logcdf( - term / tau_seq[t-1]) ) 
            wt[i] = constraint1 - constraint2 
        newW = newW * np.exp(wt)
        # normalize
        newW = newW / np.sum(newW)
        
        ESS.append( 1 / sum(newW**2) )
                
        # resample with weight
        index = np.random.choice(np.arange(0,N,1), size=N, p=newW, replace=True )
        newsample = newsample[index]
        newlpden = newlpden[index]
        newW = np.ones(N)*(1.0/N)
        
        q = np.std(newsample, axis=0) / (t+1)**qt
        a = np.vstack((a, np.zeros(dim)))
        
        # sample from K^t
        for j in range(M):
            for i in range(N):
                for d in range(dim):
                    newsample[i], a[t,d], newlpden[i] = Gibbs(newsample[i], q, d, a[t,d], tau_seq[t], newlpden[i], constraint_func)   
        
        sample_seq.append( newsample )
        lpden_seq.append( newlpden )
        W_seq.append( newW ) 
        
        if tau_seq[t] <= tau_T:
            break
    
    sample = sample_seq[t]
    return sample