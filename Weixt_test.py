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
    color = color[0]
    ax.plot([n, n], [min_time, max_time], color)
    if n < 20:
        ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    return mean_time


numIter = 10
N=50
A_input = []
if 0:
    # A_input = np.array([[.1, 0], [-.1, 0], [1, 0]])  #,[0, -1] ]) # A_input is a matrix containing the sampled points a_i
    A_input = np.array([[1, 0], [-1, 0], [0, 1]])
    #A_input = np.array(data['Q'])
else:
    for i in range(N):
        temp = np.random.uniform(-1, 1, 2)
        if temp[0] >0 and temp[1] > 0:
          continue
        else:
            A_input.append(temp)
A_input = np.array(A_input)
Point_coordinate_sum = A_input.sum(axis=0)
x_0 = Point_coordinate_sum / N
a_median_weisz, a_steps_weisx = weisz_calc(A_input, numIter)
b_median_weisz, b_steps_weisx = weisz_calc(A_input, int(numIter/2))


plt.plot(A_input[:, 0], A_input[:, 1], 'ro')
plt.plot(a_median_weisz[0],a_median_weisz[1], 'ks')
plt.plot(b_median_weisz[0],b_median_weisz[1], 'k*')
a_steps_weisx = np.array(a_steps_weisx)
plt.plot(a_steps_weisx[:,0],a_steps_weisx[:,1], 'g4')
plt.plot(x_0[0],x_0[1], 'b*')
plt.show()
print(a_median_weisz)



