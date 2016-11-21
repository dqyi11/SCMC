'''
Created on Nov 15, 2016

@author: daqingy
'''
   
import numpy as np
import matplotlib.pyplot as plt
from scmc import *

# x**2 + y**2 = 1

def on_circle_func(sample):    
    out = np.zeros(1)
    out[0] = np.abs( sample[0]**2 + sample[1]**2 - 1 )
    return out

if __name__ == '__main__':
    
    srng0 = [[-2,2],[-2,2]]
    RV_X = UniformRandomVariable(2, srng0)  
    sample0 = scmc(RV_X, N=1000, M=10, constraint_func=on_circle_func, tau_T= 1e3)
    
    circle1 = plt.Circle((0,0),1,color='r', fill=False) 
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.add_artist(circle1)
    ax1.scatter(sample0[:,0],sample0[:,1])
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    
    plt.show()