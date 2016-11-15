'''
Created on Nov 15, 2016

@author: daqingy
'''
   
import numpy as np
import matplotlib.pyplot as plt
from scmc import *

# y = log(20x+1)

def func1(x):
    y = np.log(20*x+1)
    return y

def on_func1(sample):    
    out = np.zeros(1)
    out[0] = func1(sample[0]) - sample[1] 
    return out

if __name__ == '__main__':
    
    srng0 = [[0,1],[0,2]]
    #sample0 = scmc(N=1000, dim=2, M=20, L=25, srng=srng0, constraint_func=on_func1, tau_T= 1e-3, qt = 1)
    
    X = np.arange(0.0, 1.0, 0.01)
    Y = func1(X)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(X, Y, color='r')
    #ax1.scatter(sample0[:,0],sample0[:,1], color='b')
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    
    plt.show()