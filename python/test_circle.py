'''
Created on Nov 15, 2016

@author: daqingy
'''

import numpy as np
import matplotlib.pyplot as plt
from scmc import *

# x**2 + y**2 = 1

def circle_func(sample):    
    out = np.zeros(2)
    #out[0] = sample[0]**2 + sample[1]**2 - 1
    if np.abs(sample[0]) > 1:
        out[0] = 10
        out[1] = 10
    else:
        out[0] = np.sqrt(1-sample[0]**2) - sample[1]
        out[1] = - np.sqrt(1-sample[1]**2) - sample[1]
    return out

if __name__ == '__main__':
    
    srng0 = [[-1,1],[-1,1]]
    sample0 = scmc(N=1000, dim=2, M=10, L=25, srng=srng0, constraint_func=circle_func, tau_T= 1e-3, qt = 1)
    
    circle1 = plt.Circle((0,0),1,color='r', fill=False)   
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.add_artist(circle1)
    ax1.scatter(sample0[:,0],sample0[:,1],color='b')
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    ax1.set_xlim((-1,1))
    ax1.set_ylim((-1,1))
    
    plt.show()