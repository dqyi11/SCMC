'''
Created on Nov 17, 2016

@author: walter
'''

import numpy as np
import matplotlib.pyplot as plt
from scmc import *

# (x-2)**2 + y**2 = 1
# (x+2)**2 + y**2 = 1

def two_circles_func(sample):    
    out = np.zeros(2)
    out[0] = min( (sample[0]-2) **2 + sample[1]**2 - 1, (sample[0]+2) **2 + sample[1]**2 - 1 )
    return out

if __name__ == '__main__':
    
    #srng0 = [[-4,4],[-1,1]]
    RV_X = NormalRandomVariable(2, [0.,0.], [.5,.5])  
    sample0 = scmc(RV_X, N=500, M=10, constraint_func=two_circles_func, tau_T= 1e3)
    
    circle1 = plt.Circle((-2,0),1,color='r', fill=False)   
    circle2 = plt.Circle((2,0),1,color='r', fill=False)   
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.add_artist(circle1)
    ax1.add_artist(circle2)
    ax1.scatter(sample0[:,0],sample0[:,1],color='b')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_xlim((-4,4))
    ax1.set_ylim((-1,1))
    
    plt.show()