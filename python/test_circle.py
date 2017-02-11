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

def circle_func(sample):    
    out = np.zeros(1)
    out[0] = sample[0]**2 + sample[1]**2 - 1
    return out

def log_circle_func(sample, tau_t, RV_X):
    
    circle_term = circle_func(sample)
    term = np.sum( norm.logcdf(- circle_term * tau_t) ) + RV_X.logpdf(sample)
    return term

if __name__ == '__main__':
    
    srng0 = [[-1,1],[-1,1]]
    #RV_X = UniformRandomVariable(2, srng0)
    RV_X = NormalRandomVariable(2, [0.,1.], [.5,.5])  
    sample0, W0, lpden0  = scmc(RV_X, N=1000, M=10, log_constraint_func=log_circle_func, tau_T= 1e3)
    sample1 = rejection_sampling(RV_X, 1000, constraint_func=circle_func)
    
    print ks_test(sample0, sample1)
    
    circle1 = plt.Circle((0,0),1,color='r', fill=False)
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.add_artist(circle1)
    ax1.scatter(sample0[:,0],sample0[:,1],color='b')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_xlim((-1,1))
    ax1.set_ylim((-1,1))
    ax1.set_title('SCMC')
    
    fig2 = plt.figure()
    ax2 =fig2.add_subplot(111)
    ax2.add_artist(circle1)
    ax2.scatter(sample1[:,0],sample1[:,1],color='b')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim((-1,1))
    ax2.set_ylim((-1,1))
    ax2.set_title('Rejection Sampling')
    
    plt.show()