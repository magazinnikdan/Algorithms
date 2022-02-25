import numpy as np
from numpy import linalg as LA
import matplotlib.image as mpimg
import matplotlib as mpl
import Eng_module as eng
import time
import matplotlib.pyplot as plt
import pickle
from mat4py import loadmat
import math
data = loadmat('mat(10).mat')

res =0.01
n_dist_mat = int(401*0.01/res);

if 1:
    dist_mat=np.zeros((n_dist_mat,n_dist_mat))
    center_dist_mat =np.floor(n_dist_mat/2)
    for i in range(n_dist_mat):
        for j in range(n_dist_mat):
            dist_mat[i,j] = (res*((center_dist_mat-i)**2+(center_dist_mat-j)**2)**0.5)#.astype(np.int16)
else:
    dist_mat = pickle.load(open("save.p", "rb"))
    center_dist_mat =np.floor(n_dist_mat/2)
pickle.dump(dist_mat, open("save_0p01.p", "wb"))
# print('done')

def line_dist_calc(A_input, p1, p2):
    line_dist = 0
    for j in range(len(A_input[:, 1])):
        p3 = np.array(A_input[j])
        d = LA.norm(np.cross(p2 - p1, p1 - p3)) / LA.norm(p2 - p1)
        line_dist += d
    return line_dist


def draw_statistical_lines(A_input,n_lines):
    line_dist_min_s = 10000
    p1_best_s = []
    p2_best_s = []
    for i in range(n_lines):
        p1 = np.array(np.random.uniform(-2, 2, D))
        p2 = np.array(np.random.uniform(-2, 2, D))
        line_dist_s = line_dist_calc(A_input, p1, p2)
        if line_dist_s < line_dist_min_s:
            line_dist_min_s = line_dist_s
            p1_best_s = p1
            p2_best_s = p2
    plt.plot([p1_best_s[0], p2_best_s[0]], [p1_best_s[1], p2_best_s[1]],'r', linewidth=3)
    print("Dist_Stat= ", line_dist_min_s)


D = 2
if 1:
    # A_input = np.array([[.1, 0], [-.1, 0], [1, 0]])  #,[0, -1] ]) # A_input is a matrix containing the sampled points a_i
    A_input = np.array([[0, 0.6], [0, 0.6], [0, -0.4], [0, -0.4]])
    N = len(A_input[:,1])
    #A_input = np.array(data['Q'])
else:
    N = 4
    s = (N, D)
    A_input=np.zeros(s)
    for i in range(N):
            A_input[i]=  np.random.uniform(-2, 2, D)
dist_mat_saved = pickle.dump( dist_mat, open( "save.p", "wb" ) )


max_input=abs(A_input).max()#k
n_input_dist=round((2*max_input)/res)#m
#dist_mat_sum=np.zeros((n_input_dist,n_input_dist))
center_input_dist = (n_input_dist/2)#c
#dist_mat_total=np.zeros((n_input_dist,n_input_dist))



dist_mat = pickle.load(open("save.p", "rb"))
center_dist_mat =np.floor(n_dist_mat/2)

P_mesh=(int(center_input_dist))
x = np.arange(P_mesh, -P_mesh,-1)
y = np.arange(-P_mesh, P_mesh,1)

tic = time.perf_counter()
for i in range(len(A_input[:,1])):
    x_i=A_input[i,:]
    ind_i_min = round(center_dist_mat+(x_i[0]/res)-n_input_dist/2)
    ind_i_max = round(center_dist_mat+(x_i[0]/res)+n_input_dist/2)
    ind_j_min = round(center_dist_mat-(x_i[1]/res)-n_input_dist/2)
    ind_j_max = round(center_dist_mat-(x_i[1]/res)+n_input_dist/2)
    dist_mat_total=dist_mat[ind_j_min:ind_j_max,ind_i_min:ind_i_max]
    if i == 0:
        dist_mat_sum = dist_mat_total
    else:
        dist_mat_sum=dist_mat_sum+dist_mat_total

toc = time.perf_counter()
plt.plot(A_input[:, 0], A_input[:, 1], 'ro')
print("Computation time",toc-tic)

arg = np.argmin(dist_mat_sum)

P_mesh=len(dist_mat_sum[1,:])/2
x = np.arange(P_mesh, -P_mesh,-1)
y = np.arange(-P_mesh, P_mesh,1)
h = plt.contourf(x*res, y*res, dist_mat_sum)
plt.plot((center_input_dist-arg%n_input_dist)*res,-(center_input_dist-arg//n_input_dist)*res, 'ks')
t_0=0.001
dist_dif=100000
Point_coordinate_sum = A_input.sum(axis=0)
x_0 = Point_coordinate_sum / N
a_median_eng=eng.find_median(t_0,A_input,x_0)
plt.plot(a_median_eng[0],a_median_eng[1], 'g4')
median = [(center_input_dist-arg%n_input_dist)*res, -(center_input_dist-arg//n_input_dist)*res]
print(f"The Eng median is {a_median_eng}")
print(f'The median is {median}')
mean_f=eng.f(A_input,x_0)
a_median_f=eng.f(A_input,median)
a_median_eng_f=eng.f(A_input,a_median_eng)

print(f"The median ratio is: {a_median_eng_f/a_median_f} and the mean ratio is {a_median_eng_f/mean_f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('scaled')
calc_lines = 1
tic = time.perf_counter()
if calc_lines:
    line_dist_min = math.inf
    p1_best=[]
    p2_best=[]
    theta_best = []
    n_lines = 180
    line_dist_diff = 1000
    t_line = 0.1
    for k in range(int(200/N)):
        line_dist_diff = 1000
        #line_dist_min = math.inf
        p1 = np.random.uniform(-0.1, 0.1, D)
        for i in range(n_lines):
            p2 = p1 + np.array([2 * math.sin(i / n_lines * math.pi), 2 * math.cos(i / n_lines * math.pi)])
            line_dist = 0
            line_dist = line_dist_calc(A_input, p1, p2)
            if line_dist < line_dist_min:
                line_dist_min = line_dist
                p1_best = p1
                p2_best = p2
                theta_best = i
        flag = 1
        while line_dist_diff > 0:
            p1 = p1_best + t_line * np.array([math.sin(theta_best/180*math.pi),math.cos(theta_best/180*math.pi)])
            p2 = p2_best + t_line * np.array([math.sin(theta_best / 180 * math.pi),math.cos(theta_best / 180 * math.pi)])
            line_dist = line_dist_calc(A_input, p1, p2)
            line_dist_diff = line_dist_min - line_dist
            if line_dist_diff < 0 and flag == 1:
                t_line = t_line * (-1)
                line_dist_diff = 1000
                flag = 0
            elif line_dist_diff > 0:
                p1_best = p1
                p2_best = p2
                line_dist_min = line_dist
            #plt.plot([p1_best[0], p2_best[0]], [p1_best[1], p2_best[1]], linewidth=1)


        #print("Dist= ", line_dist_min)
        n_lines_s = 10000
        toc = time.perf_counter()
    print("Dist= ", line_dist_min)
    plt.plot([p1_best[0], p2_best[0]], [p1_best[1], p2_best[1]], linewidth=2)
    plt.plot(p1_best[0], p1_best[1], 'b4')
    plt.plot(p2_best[0], p2_best[1], 'b4')
    print("Line computation time", toc - tic)
    tic = time.perf_counter()
    draw_statistical_lines(A_input, n_lines_s)
    toc = time.perf_counter()
    print("Line computation time stat", toc - tic)

# tic = time.perf_counter()
# dist_mat_coord=dist_mat_sum
# for i in range(n_input_dist):
#     for j in range(n_input_dist):
#         dist_a_l=0
#         for l in range(N):
#             dist_a_l += res*((center_dist_mat-i)**2+(center_dist_mat-j)**2)**0.5
#             #dist_a_l += LA.norm(x_0 - A_input[l, :], ord=2, axis=None, keepdims=False)
#         dist_mat_coord[i,j] = dist_a_l
# toc = time.perf_counter()
# print("Loop computation time",toc-tic)
plt.show()

