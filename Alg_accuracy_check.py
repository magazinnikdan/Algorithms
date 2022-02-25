import light_median_calc as lmc
import Eng_module as eng
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
#from scipy.io import loadmat
import time
from mat4py import loadmat
from Weiszfeld_DJL import weisz_calc


def jap_can_dan(n, time_vec, color, label, ax):
    time_vec = np.array(time_vec)
    mean_time = time_vec.sum()/len(time_vec)
    max_time = np.max(time_vec)
    min_time = np.min(time_vec)
    ax.plot([n], [mean_time], f'{color}', label=label)
    color=color[0]
    ax.plot([n, n], [min_time, max_time], color)
    if n < 20:
        ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    return mean_time


res = 0.01
K = 1
N = 500
error_vec = []
numIter = 10

for l in range(K):
    if 0:
        # A_input = np.array([[.1, 0], [-.1, 0], [1, 0]])  #,[0, -1] ]) # A_input is a matrix containing the sampled points a_i
        A_input = np.array([[-0.625, -.7608], [-.1160, -.128]])
        #A_input = np.array(data['Q'])
    else:
        s = (N, 2)
        A_input = np.zeros(s)
        for i in range(N):
            A_input[i] = np.random.uniform(-1, 1, 2)
    # v_addded = [1,1]# np.random.uniform(-1, 1, 2)
    v_addded = (np.random.uniform(-1, 1, 2))*1
    # A_input = np.concatenate((A_input, -A_input), axis=0)
    # A_input = A_input+v_addded
    tic_eng = time.perf_counter()
    a_median = np.array(lmc.median_calc(A_input, res))
    toc_eng = time.perf_counter()
    a_median_eng = eng.find_median(res / 10, A_input, a_median)

    f_eng = eng.f(A_input, a_median_eng)

    f_added = eng.f(A_input, v_addded)
    tic_weisz = time.perf_counter()
    a_median_weisz = weisz_calc(A_input, numIter)
    toc_weisz = time.perf_counter()
    f_weisz = eng.f(A_input, a_median_weisz)
    # a_median_eng = eng.find_median(res / 10, A_input, a_median_eng)
    # a_median_eng = eng.find_median(res / 100, A_input, a_median_eng)
    # f_lmc = eng.f(A_input, a_median)
    error = LA.norm(a_median_eng - v_addded, ord=2, axis=None, keepdims=False)/res
    error_vec.append(error)
# print("Median value ", a_median)
# print("Median error norm", max(error_vec))

if K == 1:
    plt.plot(A_input[:, 0], A_input[:, 1], 'ro')
    plt.plot(a_median[0],a_median[1], 'ks')
    plt.plot(a_median_eng[0],a_median_eng[1], 'g4')
    plt.plot(a_median_weisz[0], a_median_weisz[1], 'r4')
    plt.plot(v_addded[0],v_addded[1], 'c^')
else:
    n, bins, patches = plt.hist(error_vec, density=False, facecolor='b', alpha=0.75)

print(f"Eng_time {(toc_eng - tic_eng)} and the F is: {f_eng}")
print(f"Weiszfeld_time {(toc_weisz - tic_weisz)} and the F is: {f_weisz}")
print(max(error_vec))
plt.show()
