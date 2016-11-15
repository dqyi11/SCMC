'''
Created on Nov 15, 2016

@author: daqingy
'''

import numpy as np
import matplotlib.pyplot as plt
from scmc import *

# (x/36)**2 + (y/9)**2 = 1

def ellipse2d(sample):    
    out = np.zeros(2)
    out[0] = sample[0] - 6. * np.sqrt(1 - sample[1]**2/9.)
    out[1] =  - (sample[0] + 6. * np.sqrt(1 - sample[1]**2/9.))
    return out

if __name__ == '__main__':
    
    srng0 = [[-6,6],[-3,3]]
    sample0 = scmc(N=1000, dim=2, M=10, L=25, srng=srng0, constraint_func=ellipse2d, tau_T= 1e-3, qt = 1)
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.scatter(sample0[:,0],sample0[:,1])
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    
    plt.show()