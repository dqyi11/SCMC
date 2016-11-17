'''
Created on Nov 15, 2016

@author: daqingy
'''
   
import numpy as np
import matplotlib.pyplot as plt
from scmc import *

# y = 0.1 + 0.3 * x**3 + 0.5 * x**5 + 0.7 * x**7 + 0.9 * x**9

def func1(x):
    y = 0.1 + 0.3 * x**3 + 0.5 * x**5 + 0.7 * x**7 + 0.9 * x**9
    return y

if __name__ == '__main__':
    
    srng0 = [[0,1],[0,2]]
    def on_func1(sample):    
        out = np.zeros(1)
        if sample[0] < srng0[0][0]: 
            out[0] = 10 #np.abs(sample[0] - srng0[0][0])
        elif sample[0] > srng0[0][1]:    
            out[0] = 10 #np.abs(sample[0] - srng0[0][1])
        elif sample[1] < srng0[1][0]: 
            out[0] = 10 #np.abs(sample[1] - srng0[1][0])
        elif sample[1] > srng0[1][1]:    
            out[0] = 10 #np.abs(sample[1] - srng0[1][1])    
        else:
            out[0] = np.abs(func1(sample[0]) - sample[1]) 
        return out
    
    sample0 = scmc(N=100, dim=2, M=20, srng=srng0, constraint_func=on_func1, tau_T= 1e3)
    
    X = np.arange(srng0[0][0], srng0[0][1], 0.01)
    Y = func1(X)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(X, Y, color='r')
    ax1.scatter(sample0[:,0],sample0[:,1], color='b')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    plt.show()