import random
import winsound
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import time
import math
import statistics
from random import randrange
from mat4py import loadmat
data = loadmat('mat(10).mat')

def Sum_dist(A,x):
    sum=0
    for i in range(A.shape[0]):
        sum += LA.norm(A[i, :]-x)
    return sum


def Better_direction_2D(A, x, t):
    dist_x_pos = Sum_dist(A, x+[t, 0])
    dist_x_neg = Sum_dist(A, x + [-t, 0])
    dist_y_pos = Sum_dist(A, x + [0, t])
    dist_y_neg = Sum_dist(A, x + [0, -t])
    dist_xy_pos = Sum_dist(A, x + [0.7*t, 0.7*t])
    dist_xy_neg = Sum_dist(A, x + [0.7*t, -0.7*t])
    dist_yx_pos = Sum_dist(A, x + [-0.7*t, 0.7*t])
    dist_yx_neg = Sum_dist(A, x + [-0.7*t, -0.7*t])
    dist_vec = [dist_x_pos, dist_x_neg, dist_y_pos, dist_y_neg,dist_xy_pos,dist_xy_neg,dist_yx_pos,dist_yx_neg ]
    x_vec=[x+[t, 0], x + [-t, 0], x + [0, t], x + [0, -t],x + [0.7*t, 0.7*t],x + [0.7*t, -0.7*t],x + [-0.7*t, 0.7*t],x + [-0.7*t, -0.7*t]]
    dist_dif = Sum_dist(A, x)-min(dist_vec)
    return x_vec[np.argmin(dist_vec)], dist_dif


def Better_direction_2D_rand(A, x, t,N_rand):
    dist_rand=[]
    dir_rand=[]
    for _ in range(0,N_rand):
        dir_rand.append(np.random.normal(0, t, D))
        dist_rand.append(Sum_dist(A, x + dir_rand[-1]))

    dist_dif = Sum_dist(A, x)-min(dist_rand)
    return x + dir_rand[np.argmin(dist_rand)], dist_dif


def find_median(t_0,dist_dif,A_input,x_new):
    while np.sign(dist_dif)>0 :
        t = statistics.median([dist_dif,10*t_0,t_0])
        #t=t_0
        x_new, dist_dif = Better_direction_2D(A_input, x_new, t)
        nnn=10
        #x_new, dist_dif = Better_direction_2D_rand(A_input, x_new, t, nnn)

        #t = max(dist_dif, t_0)
        axis.plot(x_new[0], x_new[1], 'g*')
        dist_vec.append(dist_dif)
    return x_new, dist_vec


t_0 = 0.01
N = 1000
D = 2
if 0 :
    A_input = np.array([[1,0], [-1, 0], [-0.5, 0]])
    #A_input = np.array(data['Q'])
else:

    s = (N, D)
    A_input=np.zeros(s)
    for i in range(N):
            #A_input[i] = np.random.normal(0, 5, D)
            A_input[i] = np.random.uniform(-2, 2, D)


n = len(A_input[:, 1])
d = len(A_input[1, :])
Point_coordinate_sum = A_input.sum(axis=0)
x_0 = Point_coordinate_sum / n#+[2, 1]
dist=Sum_dist(A_input,x_0)
#print(f"Sum of distanes from {x_0} is: {dist}.  ")


x_new, dist_dif =Better_direction_2D(A_input, x_0, t_0)

#print(f"Better location {x_new} is: {dist}. ")
#print(f"dDistance differance is: {dist_dif}")
dist_vec=[]
figure, axis = plt.subplots()
figure, axis2 = plt.subplots()
t=t_0
tic = time.perf_counter()
x_median, diff_vec=find_median(t_0,dist_dif,A_input,x_new)

toc = time.perf_counter()

# N_samp=100
# x_0_rand_vec=[]
# for _ in range(N_samp):
#     N_rand = 10
#     A_rand = A_input[random.sample(range(N), k=N_rand), :]
#     axis.plot(A_rand[:,0], A_rand[:,1], 'm+')
#     Rand_point_coordinate_sum = A_rand.sum(axis=0)
#     x_0_rand = Rand_point_coordinate_sum / N_rand
#     #axis.plot(x_0_rand[0], x_0_rand[1], 'cs')
#     x_0_rand_vec.append(x_0_rand)
# x_0_rand_vec_sum = sum(x_0_rand_vec)
# x_0_rand_average = x_0_rand_vec_sum / N_samp
# axis.plot(x_0_rand_average[0], x_0_rand_average[1], 'r*')



axis.plot(A_input[:,0], A_input[:,1], 'ro')
axis.plot(x_0[0], x_0[1], 'bs')
axis.plot(x_median[0], x_median[1], 'k*')
axis2.plot(range(len(diff_vec)), diff_vec, 'bs')
print(f"Final x is {x_median} ")
print(f"Calculated in {toc-tic} seconds")
axis.axis('equal')
plt.show()
