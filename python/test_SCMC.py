'''
Created on Nov 14, 2016

@author: daqingy
'''

import numpy as np
import matplotlib.pyplot as plt
from scmc import *

def crecsent(sample):    
    out = [sample[1] - np.sqrt(14 * sample[2] ^ 2 + 2), - (sample[1] - np.sqrt(33*sample[2] ^ 2 + 1))]
    return out

def mixture(sample, l= [.4,.1,.1,.03], u = [.6, .47, .47, .08]):
    return [abs(sum(sample) - 1), l - sample, sample - u]

if __name__ == '__main__':
    
    srng0 = []
    sample0 = scmc(N=1000, dim=2, M=10, L=25, srng=srng0, constraint_func=crecsent, tau_T= 1e-3, qt = 1)
    
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.scatter(sample0[:,0],sample0[:,1])
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    
    plt.show()
    