import light_median_calc as lmc
import Eng_module as eng
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
#from scipy.io import loadmat
import time
from michael_cohen_base import michael_median



res = 0.01
K = 1
N = 3

print('1')
for _ in range(K):
    if 0:
        # A_input = np.array([[.1, 0], [-.1, 0], [1, 0]])  #,[0, -1] ]) # A_input is a matrix containing the sampled points a_i
        A_input = np.array([[-0.625, -.7608], [-.1160, -.128]])
        #A_input = np.array(data['Q'])
    else:
        s = (N, 2)
        A_input = np.zeros(s)
        for i in range(N):
            A_input[i] = np.random.uniform(-1, 1, 2)
            print(1)
    v_addded = (np.random.uniform(-1, 1, 2))*1

print(A_input)
median = np.array(lmc.median_calc(A_input, res))
print(median)
#med = michael_median(A_input)



