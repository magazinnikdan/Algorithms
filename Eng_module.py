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
        dir_rand.append(np.random.normal(0, t, 2))
        dist_rand.append(Sum_dist(A, x + dir_rand[-1]))

    dist_dif = Sum_dist(A, x)-min(dist_rand)
    return x + dir_rand[np.argmin(dist_rand)], dist_dif


def find_median(t_0, A_input, x_new):
    dist_dif = 100000
    while np.sign(dist_dif) > 0:
        # t = statistics.median([dist_dif, 10 * t_0, t_0])
        t = t_0
        x_new, dist_dif = Better_direction_2D(A_input, x_new, t)
        # nnn=10
        # x_new, dist_dif = Better_direction_2D_rand(A_input, x_new, t, nnn)

    return x_new


def f(A_input, x):
    # Input: Sampled points' matrix nxd (A_input), point in R_d(x)
    # Output: Scalar sum of L2 norm of  x-a_i
    # page 3
    f_sum = 0
    for i in range(0, len(A_input[:, 1])):
        f_sum += LA.norm(x - A_input[i, :], ord=2, axis=None, keepdims=False)
    return (f_sum)