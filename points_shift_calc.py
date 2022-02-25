import light_median_calc as lmc
import Eng_module as eng
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


shift = [1, 1]
res = 0.01
N =5
if 0 :
    # A_input = np.array([[.1, 0], [-.1, 0], [1, 0]])  #,[0, -1] ]) # A_input is a matrix containing the sampled points a_i
    A_input = np.array([[.005, 0], [-.01, 0], [0, 0.01]])
    #A_input = np.array(data['Q'])
else:
    s = (N, 2)
    A_input=np.zeros(s)
    for i in range(N):
            A_input[i] = np.random.uniform(-1, 1, 2)

B_input = np.zeros([len(A_input[:,1]),2])
for i in range(len(A_input[:,1])):
    B_input[i] = A_input[i] + shift#+np.random.normal(0, 0.010, 2)

t_0=0.0001
dist_dif=100000
Point_coordinate_sum = A_input.sum(axis=0)
x_0 = Point_coordinate_sum / N
a_median = np.array(lmc.median_calc(A_input,res))
a_median_eng=eng.find_median(t_0,dist_dif,A_input,x_0)
print(f"The Eng median is {a_median_eng}")
print(f"The median is {a_median} and the mean is {x_0}")
mean_f=eng.f(A_input,x_0)
a_median_f=eng.f(A_input,a_median)
a_median_eng_f=eng.f(A_input,a_median_eng)

print(f"The median ratio is: {a_median_eng_f/a_median_f} and the mean ratio is {a_median_eng_f/mean_f}")

b_median = np.array(lmc.median_calc(B_input,res))
Point_coordinate_sum = B_input.sum(axis=0)
x_0 = Point_coordinate_sum / N
print(f"The median is {b_median} and the mean is {x_0}")

shift_calc = b_median-a_median
print(f"Added shift is: {shift}")
print(f"The calculated shift is:{shift_calc}, using resolution of: {res}")
B_shifted=B_input-shift_calc

plt.plot(A_input[:, 0], A_input[:, 1], 'ro')
plt.plot(B_shifted[:, 0], B_shifted[:, 1], 'bs')


plt.show()

