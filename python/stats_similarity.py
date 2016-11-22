'''
Created on Nov 22, 2016

@author: daqingy
'''

from scipy import stats
import numpy as np

def ks_test(data1, data2):
    
    dim = data1.shape[1]
    result = np.zeros((dim,2))
    
    for i in range(dim):
        ks = stats.ks_2samp(data1[:,i], data2[:,i])
        result[i,0] = ks[0]
        result[i,1] = ks[1]
    return result