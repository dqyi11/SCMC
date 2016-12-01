'''
Created on Nov 14, 2016

@author: daqingy
'''

import numpy as np
import matplotlib.pyplot as plt
from scmc import *
from rejection_sampler import *
from stats_similarity import *

def crecsent(sample):    
    out = np.zeros(2)
    out[0] = sample[0] - np.sqrt(14 * sample[1]**2 + 2)
    out[1] =  - (sample[0] - np.sqrt(33*sample[1]**2 + 1))
    return out

if __name__ == '__main__':
   
    srng0 = [[-1,1],[-1,1]]
    #RV_X = UniformRandomVariable(2, srng0)  
    RV_X = NormalRandomVariable(2, [0.,0.], [.5,.5])  
    sample0, W0, lpden0 = scmc(RV_X, N=1000, M=10, constraint_func=crecsent, tau_T= 1e3)
    sample1 = rejection_sampling(RV_X, 1000, constraint_func=crecsent)
    
    print ks_test(sample0, sample1)
    
    yl = np.sqrt(1.0/19)
    Y = np.arange(-yl, yl, 0.01)
    X1 = np.sqrt(1+33 * (Y**2))
    X2 = np.sqrt(2+14 * (Y**2))
    
    max_idx = np.argmax(lpden0)
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.plot(X1, Y, color='r')
    ax1.plot(X2, Y, color='r')
    ax1.scatter(sample0[:,0],sample0[:,1],color='b')
    ax1.plot(sample0[max_idx,0],sample0[max_idx,1],marker='o',color='r')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('SCMC')
    
    fig2 = plt.figure()
    ax2 =fig2.add_subplot(111)
    ax2.plot(X1, Y, color='r')
    ax2.plot(X2, Y, color='r')
    ax2.scatter(sample1[:,0],sample1[:,1],color='b')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Rejection Sampling')
    
    plt.show()
    