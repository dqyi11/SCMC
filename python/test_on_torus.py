'''
Created on Nov 15, 2016

@author: daqingy
'''
   
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scmc import *
from rejection_sampler import *
from stats_similarity import *
import math

# (2 - sqrt(x**2+y**2))**2 + z**2 = 1

def on_torus_func(sample):    
    out = np.zeros(1)
    out[0] = abs( (2 - math.sqrt(sample[0]**2+sample[1]**2))**2 + sample[2]**2 - 1 )
    return out

if __name__ == '__main__':
    
    #srng0 = [[-1,1],[-1,1],[-1,1]]
    srng0 = [[-4,4],[-4,4],[-4,4]]
    
    #RV_X = UniformRandomVariable(3, srng0)  
    RV_X = NormalRandomVariable(3, [0.,1., .5], [1,1,1])  
    sample0, W0, lpden0 = scmc(RV_X, N=10000, M=10, constraint_func=on_torus_func, tau_T= 1e4)
    '''
    sample1 = rejection_sampling(RV_X, 5000, constraint_func=on_torus_func, tolerance=1e-4)
    print ks_test(sample0, sample1)
    '''
    max_idx = np.argmax(lpden0)
    print "MAX IDX:" + str(max_idx)
    print "MAX:" + str(sample0[max_idx,:])
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111, projection='3d')
    ax1.scatter(sample0[:max_idx-1,0],sample0[:max_idx-1,1],sample0[:max_idx-1,2],color='b')
    ax1.scatter(sample0[max_idx+1:,0],sample0[max_idx+1:,1],sample0[max_idx+1:,2],color='g')
    ax1.scatter(sample0[max_idx,0],sample0[max_idx,1],sample0[max_idx,2],color='r')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('SCMC')
    
    '''
    fig2 = plt.figure()
    ax2 =fig2.add_subplot(111, projection='3d')
    ax2.scatter(sample1[:,0],sample1[:,1],sample1[:,2])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Rejection Sampling')
    '''
    plt.show()