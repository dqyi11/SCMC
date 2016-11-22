'''
Created on Nov 15, 2016

@author: daqingy
'''

import numpy as np
import matplotlib.pyplot as plt
from scmc import *
from rejection_sampler import *
from stats_similarity import *
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
    #RV_X = UniformRandomVariable(2, srng0)
    RV_X = NormalRandomVariable(2, [0.,1.], [2.,2.])  
    sample0, W0, lpden0 = scmc(RV_X, N=1000, M=10, constraint_func=ellipse2d, tau_T= 1e3)
    sample1 = rejection_sampling(RV_X, 1000, constraint_func=ellipse2d, tolerance=0)
    
    max_idx = np.argmax(lpden0)
    
    print ks_test(sample0, sample1)
    
    e = Ellipse(xy=[0,0], width=6, height=12, angle=-(90-rotate_angle*180/np.pi))
    e.set_fill(False)
    e.set_color('r')
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)   
    ax1.add_artist(e)
    ax1.scatter(sample0[:,0],sample0[:,1])
    ax1.plot(sample0[max_idx,0],sample0[max_idx,1],marker='o',color='r')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('SCMC')
    
    fig2 = plt.figure()
    ax2 =fig2.add_subplot(111)   
    ax2.add_artist(e)
    ax2.scatter(sample1[:,0],sample1[:,1])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Rejection Sampling')    
    
    plt.show()