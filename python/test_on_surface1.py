'''
Created on Nov 17, 2016

@author: walter
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scmc import *


# (2 - sqrt(x**2+y**2))**2 + z**2 = 1

if __name__ == '__main__':
    
    srng0 = [[-1,1],[0,np.pi],[-1,1]]
    
    def on_surface1_func(sample):    
        out = np.zeros(7)
        out[0] =  srng0[0][0] - sample[0]
        out[1] =  sample[0] - srng0[0][1]  
        out[2] = srng0[1][0] - sample[1]
        out[3] = sample[1] - srng0[1][1]  
        out[4] = srng0[2][0] - sample[2]
        out[5] = sample[2] - srng0[2][1]  
        out[6] = np.abs( sample[2] - sample[0] * np.cos(sample[1]) )
        return out
    
    sample0 = scmc(N=1000, dim=3, M=10, srng=srng0, constraint_func=on_surface1_func, tau_T= 1e4)
    
    X = np.arange(-1,1,0.01)
    Y = np.arange(0, np.pi, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = X * np.cos(Y)
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, Z, alpha=0.1)
    ax1.scatter(sample0[:,0],sample0[:,1],sample0[:,2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    plt.show()