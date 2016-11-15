'''
Created on Nov 15, 2016

@author: daqingy
'''
   
import numpy as np
import matplotlib.pyplot as plt
from scmc import *
import math

# (2 - sqrt(x**2+y**2))**2 + z**2 = 1

def on_torus_func(sample):    
    out = np.zeros(1)
    out[0] = abs( (2 - math.sqrt(sample[0]**2+sample[1]**2))**2 + sample[2]**2 - 1 )
    return out

if __name__ == '__main__':
    
    srng0 = [[-1,1],[-1,1]]
    sample0 = scmc(N=100000, dim=3, M=20, L=25, srng=srng0, constraint_func=on_torus_func, tau_T= 1e-3, qt = 1)
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.scatter(sample0[:,0],sample0[:,1])
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    
    plt.show()