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
        print i
        print srng
        X[:,i] = srng[i][0] + (srng[i][1]-srng[i][0])*X[:,i]
    return X

# log-posterior (equivalent to the constraint if a uniform sample is drawn)
def log_posterior(sample, tau_t, constraint_func):
    
    term = constraint_func(sample)
    return np.sum( np.log( norm.cdf(- term / tau_t) ) ) 

def Gibbs(x, q, d, a, nu_t, lpden):
    
    delta = np.random.normal(loc=0, scale=q[d], size=1)
    newx = x
    newx[d] = newx[d] + delta
    lpnum = log_posterior(newx, nu_t)
    ratio = lpnum - lpden
    prob = np.min([1, np.exp(ratio)])
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
        term = constraint_func(sample[i,])
        cons1 = np.sum( np.log( norm.cdf(- term / tau) ) )
        cons2 = np.sum( np.log( norm.cdf(- term / tau_prev)) )
        wt[i] = cons1 - cons2
  
    Wt = Wt * np.exp(wt)
    Wt = Wt / np.sum(Wt)
    
    ESS = 0.0
    if np.sum(np.isnan(Wt)) != N:
        ESS = 1/np.sum(Wt**2)
    return(ESS - (N / 2))

def scmc(N, dim, M, L, srng, constraint_func, tau_T= 1e-3, qt = 1):
    
    sample_seq = []
    lpden_seq = []
    W_seq = []
    tau_seq = [np.inf]
    b = np.arange(1.5,.1,L)
    a = np.zeros((L,dim))
    tau_seq_0 = np.hstack(([np.inf], b**7))
    
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
        
        newsample = sample_seq[t-1]
        newlpden = lpden_seq[t-1]
        newWt = W_seq[t-1]
        
        if adapt_seq(tau_T, tau_seq[t-1], t, N, newsample, newWt, constraint_func) > 0:
            tau_seq[t] = tau_T 
        else:
            lb = tau_T
            if t < 3:
                lb = np.min([.1, tau_seq_0[t]])
            ub = 1e5
            if t!= 2:
                ub = tau_seq_0[t-1]   
                
            tau_seq[t] = optimize.brenth(adapt_seq, lb, ub, args= (tau_seq[t-1], t, N, newsample, newWt, constraint_func))
                
        wt = np.zeros(N)
        for i in range(N):
            term = constraint_func(newsample[i])
            constraint1 = np.sum( np.log( norm.cdf( - term / tau_seq[t] )  ) )
            constraint2 = np.sum( np.log( norm.cdf( - term / tau_seq[t-1]) ) )
            wt[i] = constraint1 - constraint2 
        newWt = newWt * np.exp(wt)
        # normalize
        newWt = newWt / np.sum(newWt)
        
        ESS.append( 1 / sum(newWt**2) )
        
        # resample with weight
        index = np.random.choice(np.arange(0,N,1), size=N, p=newWt, replace=True )
        newsample = newsample[index]
        newlpden = newlpden[index]
        newWt = np.ones(N)*(1.0/N)
        
        q = np.std(newsample) / t**qt
        a.append(np.zeros(D))
        
        # sample from K^t
        for j in range(M):
            for i in range(N):
                for d in range(D):
                    newsample[i], a[t][d], newlpdent[i] = Gibss(newsample[i], q, d, a[t,d], tau_seq[t], lpdent=newlpden[i])   
        
        sample_seq.append( new_sample )
        lpden_seq.append( newlpden )
        W_seq.append( newWt ) 
        
        if tau_seq[t] <= tau_T:
            break
    
    sample = sample_seq[t]
    return sample