'''
Created on Nov 14, 2016

@author: daqingy
'''

import numpy as np
import matplotlib.pyplot as plt
from scmc import *

def crecsent(sample):    
    out = np.zeros(2)
    out[0] = sample[0] - np.sqrt(14 * sample[1]**2 + 2)
    out[1] =  - (sample[0] - np.sqrt(33*sample[1]**2 + 1))
    return out

if __name__ == '__main__':
   
    srng0 = [[-1,1],[-1,1]]
    sample0 = scmc(N=1000, dim=2, M=10, srng=srng0, constraint_func=crecsent, tau_T= 1e3)
    
    yl = np.sqrt(1.0/19)
    Y = np.arange(-yl, yl, 0.01)
    X1 = np.sqrt(1+33 * (Y**2))
    X2 = np.sqrt(2+14 * (Y**2))
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.plot(X1, Y, color='r')
    ax1.plot(X2, Y, color='r')
    ax1.scatter(sample0[:,0],sample0[:,1],color='b')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    plt.show()
    