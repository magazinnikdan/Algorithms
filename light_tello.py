import numpy as np
import matplotlib.image as mpimg
import matplotlib as mpl
import Eng_module as eng
import time
import matplotlib.pyplot as plt
import pickle
from mat4py import loadmat

W = 360
H = 240
D = 2

res = 1
n_dist_mat = W*3#s

if 0:
    dist_mat=np.zeros((n_dist_mat,n_dist_mat))
    center_dist_mat =np.floor(n_dist_mat/2)
    for i in range(n_dist_mat):
        for j in range(n_dist_mat):
            dist_mat[i,j] = res*((center_dist_mat-i)**2+(center_dist_mat-j)**2)**0.5
else:
    dist_mat = pickle.load(open("tello_dist_matrix.p", "rb"))
    center_dist_mat =np.floor(n_dist_mat/2)

# pickle.dump(dist_mat, open("tello_dist_matrix_025.p", "wb"))
#
if 0:
    # A_input = np.array([[.1, 0], [-.1, 0], [1, 0]])  #,[0, -1] ]) # A_input is a matrix containing the sampled points a_i
    A_input = np.array([[10, 0], [20, 0], [30, 0], [56, 0], [75, 0]])
    N = 100
    #A_input = np.array(data['Q'])
else:
    N =100
    s = (N, D)
    #A_input=np.zeros(s)
    A_input=[]
    for i in range(N):
            x = int(np.random.randint(1, W, 1))
            
            y = int(np.random.randint(1, H, 1))
            A_input.append([x, y])


#dist_mat = pickle.load(open("save.p", "rb"))
#center_dist_mat =round(n_dist_mat/2)

# P_mesh=len(dist_mat_sum[1,:])
P_mesh=W
x = np.arange(0, W,1)
y = np.arange(0, H,1)
tic = time.perf_counter()

for i in range(N):
    ind_i_min = (int(W*3/2)-(A_input[i][0]))
    ind_i_max = (int(W*3/2)-(A_input[i][0])+W)
    ind_j_min = (int(W*3/2)-(A_input[i][1]))
    ind_j_max = (int(W*3/2)-(A_input[i][1])+H)
    dist_mat_total = dist_mat[ind_j_min:ind_j_max, ind_i_min:ind_i_max]
    if i == 0:
        dist_mat_sum = dist_mat_total
    else:
        dist_mat_sum = dist_mat_sum+dist_mat_total
    # plt.plot(A_input[:, 0], A_input[:, 1], 'ro')
    # h = plt.contourf(x * res, y * res, dist_mat_total)
    # plt.show(block=True)
    #plt.pause(1)
toc = time.perf_counter()
print("Computation time",toc-tic)
print("Possible FPS",1/(toc-tic))

arg = np.argmin(dist_mat_sum)
median=[(arg%W)*res,(arg//W)*res ]
h = plt.contourf(x*res, y*res, dist_mat_sum[0:H,0:W])
plt.plot(median[0],median[1], 'ks')
t_0 = 0.3


A_input = np.array(A_input)
Point_coordinate_sum = A_input.sum(axis=0)
x_0 = Point_coordinate_sum / N
tic = time.perf_counter()
a_median_eng=eng.find_median(t_0,A_input ,x_0)
toc = time.perf_counter()
print("Computation time Eng",toc-tic)
print("Possible FPS",1/(toc-tic))
print(f"The Eng median is {round(a_median_eng[0]),round(a_median_eng[1])}")
print(f'The median is {median}')


mean_f=eng.f(A_input,x_0)
a_median_f=eng.f(A_input,median)
a_median_eng_f=eng.f(A_input,a_median_eng)
plt.plot(A_input[:, 0], A_input[:, 1], 'ro')
plt.plot(a_median_eng[0],a_median_eng[1], 'g4')
print(f"The median ratio is: {a_median_eng_f/a_median_f} and the mean ratio is {a_median_eng_f/mean_f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('scaled')
plt.show()


