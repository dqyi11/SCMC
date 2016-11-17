'''
Created on Nov 16, 2016

@author: daqingy
'''

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    X = np.arange(-10.0, 10.0, 0.01)
    Y0 = stats.norm.cdf(-0.01*X)
    Y1 = stats.norm.cdf(-0.1*X)
    Y2 = stats.norm.cdf(-1*X)
    Y3 = stats.norm.cdf(-5*X)
    Y4 = stats.norm.cdf(-100*X)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(X, Y0, linewidth=4, color='r')
    ax.plot(X, Y1, linewidth=4, color='g')
    ax.plot(X, Y2, linewidth=4, color='b')
    ax.plot(X, Y3, linewidth=4, color='c')
    ax.plot(X, Y4, linewidth=4, color='m')
    ax.legend([r'$\tau=0.01$', r'$\tau=0.1$',r'$\tau=1$',r'$\tau=5$',r'$\tau=100$'],prop={'size':20})
    ax.set_ylim([-0.02, 1.02])
    
    plt.show()
    
    