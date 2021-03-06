'''
Created on Nov 14, 2016

@author: daqingy
'''

from scmc import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    
    RV_X1 = UniformRandomVariable(2, [[-1.0, 1.0], [-5.0, 5.0]])
    X1 = RV_X1.sample(1000)
    fig1 = plt.figure()
    ax1 =fig1.add_subplot(111)
    ax1.scatter(X1[:,0],X1[:,1])
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    
    RV_X2 = UniformRandomVariable(3, [[-1.0, 1.0], [-5.0, 5.0], [-10.0, 10.0]])  
    X2 = RV_X2.sample(1000)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(X2[:,0],X2[:,1],X2[:,2])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.show()