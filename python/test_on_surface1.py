'''
Created on Nov 17, 2016

@author: walter
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scmc import *


# (2 - sqrt(x**2+y**2))**2 + z**2 = 1

def on_surface1_func(sample):    
    out = np.zeros(1)
    out[0] = np.abs( sample[2] - sample[0] * np.cos(sample[1]) )
    return out

if __name__ == '__main__':
    
    srng0 = [[-1,1],[0,np.pi],[-1,1]]
    sample0 = scmc(N=5000, dim=3, M=10, srng=srng0, constraint_func=on_surface1_func, tau_T= 1e4)
    
    X = np.arange(-1,1,0.01)
    Y = np.arange(0, 2*np.pi, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = X * np.cos(Y)
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, Z, alpha=0.5, color='r')
    ax1.scatter(sample0[:,0],sample0[:,1],sample0[:,2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    plt.show()