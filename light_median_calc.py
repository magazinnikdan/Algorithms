import numpy as np
import time
import pickle
from mat4py import loadmat
import matplotlib.pyplot as plt
import cProfile, pstats, io


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        # sortby = 'cumulative'
        sortby = 'time'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

@profile
def median_calc(A,res):

    if res == 0.01:
        #dist_mat = np.load('save_0p01.p', allow_pickle=True, mmap_mode='r')
        dist_mat = pickle.load(open("save_0p01.p", "rb"))
        # dist_mat = np.load('dist.npy')
        n_dist_mat = len(dist_mat[:, 1])
        center_dist_mat = np.floor(n_dist_mat / 2)
    elif res == 0.005:
        dist_mat = pickle.load(open("save_0p005.p", "rb"))
        n_dist_mat = len(dist_mat[:, 1])
        center_dist_mat = np.floor(n_dist_mat / 2)
    elif res == 0.001:
        dist_mat = pickle.load(open("save_0p001.p", "rb"))
        n_dist_mat = len(dist_mat[:, 1])
        center_dist_mat = np.floor(n_dist_mat / 2)
    elif res == 0.1:
        dist_mat = pickle.load(open("save_0p1.p", "rb"))
        n_dist_mat = len(dist_mat[:, 1])
        center_dist_mat = np.floor(n_dist_mat / 2)
    elif res == 1:
        dist_mat = pickle.load(open("save_1.p", "rb"))
        n_dist_mat = len(dist_mat[:, 1])
        center_dist_mat = np.floor(n_dist_mat / 2)
    else:
        n_dist_mat = int(1001*0.01/res)
        dist_mat=np.zeros((n_dist_mat,n_dist_mat))
        center_dist_mat =np.floor(n_dist_mat/2)
        for i in range(n_dist_mat):
            for j in range(n_dist_mat):
                dist_mat[i,j] = res*((center_dist_mat-i)**2+(center_dist_mat-j)**2)**0.5

    # dist_mat = pickle.load(open("tello_dist_matrix.p", "rb"))
    # center_dist_mat = np.floor(n_dist_mat / 2)

    # np.save('dist.npy', dist_mat)



    # q_dist_mat = dist_mat[0:int(center_dist_mat+1), 0:int(center_dist_mat+1)]
    # h_dist_mat = np.flip(q_dist_mat, axis=1)[:, 1:]
    # upper_dist_mat = np.concatenate((q_dist_mat, h_dist_mat), axis=1)
    # lower_dist_mat = np.flip(upper_dist_mat, axis=0)[1:, :]
    # total_dist_mat = np.concatenate((upper_dist_mat, lower_dist_mat), axis=0)
    # #print(total_dist_mat)





    max_input = abs(A).max()
    n_input_dist = int((2*max_input)/res)
    dist_mat_sum = np.zeros((n_input_dist, n_input_dist))
    center_input_dist = np.floor(n_input_dist/2)


    for i in range(len(A[:, 1])):
        x_i = A[i, :]
        ind_i_min = int(center_dist_mat+int(x_i[0]/res)-int(n_input_dist/2))
        ind_i_max = int(center_dist_mat+int(x_i[0]/res)+int(n_input_dist/2))
        ind_j_min = int(center_dist_mat-int(x_i[1]/res)-int(n_input_dist/2))
        ind_j_max = int(center_dist_mat-int(x_i[1]/res)+int(n_input_dist/2))
        ind_diff_i = ind_i_max - ind_i_min
        ind_diff_j = ind_j_max - ind_j_min
        ind_addition_i = (len(dist_mat_sum[1, :]) - ind_diff_i)
        ind_addition_j = (len(dist_mat_sum[0, :]) - ind_diff_j)
        dist_mat_total = dist_mat[ind_j_min:ind_j_max+ind_addition_j, ind_i_min:ind_i_max+ind_addition_i]
        dist_mat_sum = dist_mat_sum+dist_mat_total

    arg = np.argmin(dist_mat_sum)
    return [(center_input_dist-arg % n_input_dist)*res, -(center_input_dist-arg//n_input_dist)*res]


def input_plot(A,color,f):
    ax = plt.plot()
    plt.plot(A[:, 0], A[:, 1], color)
    return(ax)

