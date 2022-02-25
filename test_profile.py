import time
import timeit
import light_median_calc as lmc
# import Eng_module as eng
import numpy as np
# import matplotlib.pyplot as plt
from numpy import linalg as LA
# #from scipy.io import loadmat
# import time
import michael_cohen_base as mcmc
#


res = 0.01
K = 1000
N = 100

for _ in range(K):
    if 1:
        A_input = np.array([[1, 0], [-1, 0], [0, 1]]) # A_input is a matrix containing the sampled points a_i
        #A_input = np.array([[-0.625, -.7608], [-.1160, -.128]])
        #A_input = np.array(data['Q'])
    else:
        s = (N, 2)
        A_input = np.zeros(s)
        for i in range(N):
            A_input[i] = np.random.uniform(-1, 1, 2)
    # v_addded = (np.random.uniform(-1, 1, 2))*1


# med = np.array(lmc.median_calc(A_input, res))
med = mcmc.michael_median(A_input)
print(med)
#x1 = np.array([1, 1])
# x2 = np.array([2, 2])
# tic = time.perf_counter()
# for _ in range(K):
#     #temp = ((x1[0] - x2[0]) ** 2 + (x1[0] - x2[0]) ** 2) ** 0.5
#     # temp = LA.norm(x1 - x2)
#     mcmc.eu_dist(x1, x2)
# toc = time.perf_counter()
# print(toc- tic)