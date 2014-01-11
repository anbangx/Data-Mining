'''
Created on Jan 9, 2014

@author: anbangx
'''

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    alphab = ['A', 'B', 'C', 'D', 'E', 'F']
    frequencies = [23, 44, 12, 11, 2, 10]
    
    pos = np.arange(len(alphab))
    print(pos)
    width = 1.0     # gives histogram aspect to the bar diagram
    
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(alphab)
    print(ax)
    
    plt.bar(pos, frequencies, width, color='r')
    plt.show() 