'''
Created on Nov 15, 2016

@author: daqingy
'''
   
import numpy as np
import matplotlib.pyplot as plt
from scmc import *

# y = 2 * ( 1 + exp( -10x + 5 ) )^{-1}

def func3(x):
    y = 2 / ( 1+np.exp(-10*x+5) )
    return y

if __name__ == '__main__':
    
    srng0 = [[0,1],[0,2]]
    
    def on_func3(sample):    
        out = np.zeros(5)
        out[0] =  srng0[0][0] - sample[0]
        out[1] =  sample[0] - srng0[0][1]  
        out[2] = srng0[1][0] - sample[1]
        out[3] = sample[1] - srng0[1][1]  
        out[4] = np.abs(func3(sample[0]) - sample[1]) 
        return out
    
    RV_X = UniformRandomVariable(2, srng0)  
    sample0, W0, lpden0 = scmc(RV_X, N=500, M=10, constraint_func=on_func3, tau_T= 1e3)
    
    X = np.arange(0.0, 1.0, 0.01)
    Y = func3(X)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(X, Y, color='r')
    ax1.scatter(sample0[:,0],sample0[:,1], color='b')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    plt.show()