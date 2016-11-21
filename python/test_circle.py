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
    out[0] = sample[0]**2 + sample[1]**2 - 1
    return out

if __name__ == '__main__':
    
    srng0 = [[-1,1],[-1,1]]
    RV_X = UniformRandomVariable(2, srng0)
    sample0 = scmc(RV_X, N=1000, M=10, constraint_func=circle_func, tau_T= 1e3)
    
    circle1 = plt.Circle((0,0),1,color='r', fill=False)   
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.add_artist(circle1)
    ax1.scatter(sample0[:,0],sample0[:,1],color='b')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_xlim((-1,1))
    ax1.set_ylim((-1,1))
    
    plt.show()