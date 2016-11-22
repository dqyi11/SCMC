'''
Created on Nov 15, 2016

@author: daqingy
'''
   
import numpy as np
import matplotlib.pyplot as plt
from scmc import *
from rejection_sampler import *
from stats_similarity import *

# x**2 + y**2 = 1

def on_circle_func(sample):    
    out = np.zeros(1)
    out[0] = np.abs( sample[0]**2 + sample[1]**2 - 1 )
    return out

if __name__ == '__main__':
    
    srng0 = [[-2,2],[-2,2]]
    #RV_X = UniformRandomVariable(2, srng0)  
    RV_X = NormalRandomVariable(2, [-1.,0.], [.5,.5]) 
    sample0, W0, lpden0 = scmc(RV_X, N=1000, M=50, constraint_func=on_circle_func, tau_T= 1e3)
    sample1 = rejection_sampling(RV_X, 1000, constraint_func=on_circle_func, tolerance=1e-4)
    
    print ks_test(sample0, sample1)
    
    max_idx = np.argmax(lpden0)
    
    circle1 = plt.Circle((0,0),1,color='r', fill=False) 
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.add_artist(circle1)
    ax1.scatter(sample0[:,0],sample0[:,1])
    ax1.plot(sample0[max_idx,0],sample0[max_idx,1],marker='o',color='r')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('SCMC')
    
    
    circle2 = plt.Circle((0,0),1,color='r', fill=False) 
    fig2 = plt.figure()
    ax2 =fig2.add_subplot(111)
    ax2.add_artist(circle2)
    ax2.scatter(sample1[:,0],sample1[:,1])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Rejection Sampling')
    
    
    plt.show()