'''
Created on Nov 15, 2016

@author: daqingy
'''
   
import numpy as np
import matplotlib.pyplot as plt
from scmc import *

# y = log(20x+1)

def func2(x):
    y = np.log(20*x+1)
    return y


if __name__ == '__main__':
    
    srng0 = [[0,1],[0,2]]
    
    def on_func2(sample):    
        out = np.zeros(5)
        out[0] =  srng0[0][0] - sample[0]
        out[1] =  sample[0] - srng0[0][1]  
        out[2] = srng0[1][0] - sample[1]
        out[3] = sample[1] - srng0[1][1]  
        out[4] = np.abs(func2(sample[0]) - sample[1])
        return out
    
    sample0 = scmc(N=1000, dim=2, M=20, srng=srng0, constraint_func=on_func2, tau_T= 1e3)
    
    X = np.arange(0.0, 1.0, 0.01)
    Y = func2(X)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(X, Y, color='r')
    ax1.scatter(sample0[:,0],sample0[:,1], color='b')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    plt.show()