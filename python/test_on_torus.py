'''
Created on Nov 15, 2016

@author: daqingy
'''
   
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scmc import *
import math

# (2 - sqrt(x**2+y**2))**2 + z**2 = 1

def on_torus_func(sample):    
    out = np.zeros(1)
    out[0] = abs( (2 - math.sqrt(sample[0]**2+sample[1]**2))**2 + sample[2]**2 - 1 )
    return out

if __name__ == '__main__':
    
    #srng0 = [[-1,1],[-1,1],[-1,1]]
    srng0 = [[-4,4],[-4,4],[-4,4]]
    
    RV_X = UniformRandomVariable(3, srng0)  
    sample0 = scmc(RV_X, N=5000, M=20, constraint_func=on_torus_func, tau_T= 1e3)
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111, projection='3d')
    ax1.scatter(sample0[:,0],sample0[:,1],sample0[:,2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    plt.show()