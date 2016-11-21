'''
Created on Nov 15, 2016

@author: daqingy
'''

import numpy as np
import matplotlib.pyplot as plt
from scmc import *
from matplotlib.patches import Ellipse

# (x/36)**2 + (y/9)**2 = 1

rotate_angle = np.pi/6
def ellipse2d(sample):    
    out = np.zeros(1)
    x = np.cos(rotate_angle) * sample[0] + np.sin(rotate_angle) * sample[1]
    y = np.sin(rotate_angle) * sample[0] - np.cos(rotate_angle) * sample[1] 
    out[0] = x**2 / 36 + y**2 / 9 - 1
    return out

if __name__ == '__main__':
    
    srng0 = [[-6,6],[-6,6]]
    RV_X = UniformRandomVariable(2, srng0)  
    sample0 = scmc(RV_X, N=1000, M=10, constraint_func=ellipse2d, tau_T= 1e3)
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    
    e = Ellipse(xy=[0,0], width=6, height=12, angle=-(90-rotate_angle*180/np.pi))
    ax1.add_artist(e)
    e.set_fill(False)
    e.set_color('r')
    ax1.scatter(sample0[:,0],sample0[:,1])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    plt.show()