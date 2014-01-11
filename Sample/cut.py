'''
Created on Jan 10, 2014

@author: anbangx
'''

import pandas as pd
import numpy as np

if __name__ == '__main__':
    d = {}
    arr = np.random.randn(20)
    factor = pd.cut(arr, 4)
    print(factor)