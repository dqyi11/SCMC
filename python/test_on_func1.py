'''
Created on Nov 15, 2016

@author: daqingy
'''
   
import numpy as np
import matplotlib.pyplot as plt
from scmc import *
from rejection_sampler import *
from stats_similarity import *

# y = 0.1 + 0.3 * x**3 + 0.5 * x**5 + 0.7 * x**7 + 0.9 * x**9

def func1(x):
    y = 0.1 + 0.3 * x**3 + 0.5 * x**5 + 0.7 * x**7 + 0.9 * x**9
    return y

if __name__ == '__main__':
    
    srng0 = [[0,1],[0,2]]
    def on_func1(sample):    
        out = np.zeros(5)
        out[0] =  srng0[0][0] - sample[0]
        out[1] =  sample[0] - srng0[0][1]  
        out[2] = srng0[1][0] - sample[1]
        out[3] = sample[1] - srng0[1][1]    
        out[4] = np.abs(func1(sample[0]) - sample[1]) 
        return out
    
    RV_X = NormalRandomVariable(2, [0.5,0.5], [.5,.5])  
    #RV_X = UniformRandomVariable(2, srng0)  
    sample0, W0, lpden0 = scmc(RV_X, N=5000, M=10, constraint_func=on_func1, tau_T= 1e3)
    sample1 = rejection_sampling(RV_X, 1000, constraint_func=on_func1, tolerance=1e-3)
    
    print ks_test(sample0, sample1)
    
    max_idx = np.argmax(lpden0)
    
    X = np.arange(srng0[0][0], srng0[0][1], 0.01)
    Y = func1(X)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(X, Y, color='r')
    ax1.scatter(sample0[:,0],sample0[:,1], color='b')
    ax1.plot(sample0[max_idx,0],sample0[max_idx,1],marker='o',color='r')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('SCMC')
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(X, Y, color='r')
    ax2.scatter(sample1[:,0],sample1[:,1], color='b')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Rejection Sampling')
    
    plt.show()