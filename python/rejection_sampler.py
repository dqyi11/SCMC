'''
Created on Nov 21, 2016

@author: walter
'''
import numpy as np

def rejection_sampling(RV_X, N, constraint_func, tolerance = 0):
    
    i = 0
    samples = np.zeros((N, RV_X.dim))
    while i < N:
        ins = RV_X.sample(N-i)
        for j in range(len(ins)):
            c = constraint_func(ins[j])
            if max(c) <= tolerance:
                samples[i,:] = ins[j]
                i += 1
        print i
    return samples
    
    