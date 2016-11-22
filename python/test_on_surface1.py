'''
Created on Nov 17, 2016

@author: walter
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scmc import *
from rejection_sampler import *
from stats_similarity import *

#  z = x * cos(y)

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
    
    #RV_X = UniformRandomVariable(3, srng0)  
    RV_X = NormalRandomVariable(3, [1.,0., 1.], [.5,.5,.5])  
    sample0, W0, lpden0 = scmc(RV_X, N=5000, M=10, constraint_func=on_surface1_func, tau_T= 1e4)
    '''
    sample1 = rejection_sampling(RV_X, 1000, constraint_func=on_surface1_func, tolerance=1e-3)
    
    print ks_test(sample0, sample1)
    '''
    max_idx = np.argmax(lpden0)
    print "MAX:" + str(sample0[max_idx,:])
    
    X = np.arange(-1,1,0.01)
    Y = np.arange(0, np.pi, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = X * np.cos(Y)
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, Z, alpha=0.1)
    #ax1.scatter(sample0[:,0],sample0[:,1],sample0[:,2])
    ax1.scatter(sample0[max_idx,0],sample0[max_idx,1],sample0[max_idx,2],marker='^',color='r')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('SCMC')
    '''
    fig2 = plt.figure()
    ax2 =fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, Z, alpha=0.1)
    ax2.scatter(sample1[:,0],sample1[:,1],sample1[:,2])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Rejection Sampling')
    '''
    plt.show()