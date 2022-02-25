import light_median_calc as lmc
import Eng_module as eng
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
#from scipy.io import loadmat
import time
from mat4py import loadmat
from Weiszfeld_DJL import weisz_calc

sedumi_times = loadmat('times.mat')
sedumi_times = np.array(sedumi_times['time_total'])



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


res = 0.1
K = 1
t_0 = 0.01
numIter = 2
n_vec = [10, 50, 100, 500, 1000, 5000, 10000]
medians_sed = loadmat('medians.mat')


means_sed = []
means_lmc = []
means_eng = []
means_weisz = []

medians_lmc = []
medians_eng = []
medians_mean = []
medians_weisz = []

fig, axs = plt.subplots(1, 1, figsize=(8, 5))
fig1, axs2 = plt.subplots(1, 1, figsize=(8, 5))
for j in range(len(n_vec)):
    N = n_vec[j]
    lmc_time_vec = []
    eng_time_vec = []
    weisz_time_vec = []
    medians_lmc_vec = []
    medians_eng_vec = []
    medians_mean_vec = []
    medians_sed_vec = []
    medians_weisz_vec = []
    for l in range(K):
        if 0:
            # A_input = np.array([[.1, 0], [-.1, 0], [1, 0]])  #,[0, -1] ]) # A_input is a matrix containing the sampled points a_i
            A_input = np.array([[.005, 0], [-.01, 0], [0, 0.01]])
            #A_input = np.array(data['Q'])
        else:
            s = (N, 2)
            A_input = []
            for i in range(N):
                temp = np.random.uniform(-1, 1, 2)
                if temp[0] > 0 and temp[1] > 0:
                    continue
                else:
                    A_input.append(temp)
        A_input = np.array(A_input)
        #mat = loadmat(f'mat({n_vec[j]})({l+1}).mat')
        #A_input = np.array(mat['Q'])
        dist_dif = 100000
        Point_coordinate_sum = A_input.sum(axis=0)
        x_0 = Point_coordinate_sum / N
        tic_lmc = time.perf_counter()
        a_median = np.array(lmc.median_calc(A_input, res))
        toc_lmc = time.perf_counter()
        f_lmc = eng.f(A_input, a_median)
        medians_lmc_vec.append(f_lmc)
        lmc_time_vec.append(toc_lmc - tic_lmc)

        tic_weiz = time.perf_counter()
        a_median_weisz, steps_weisx = weisz_calc(A_input, numIter)
        f_weisz = eng.f(A_input, a_median_weisz)

        medians_weisz_vec.append(f_weisz)
        toc_weiz = time.perf_counter()
        weisz_time_vec.append(toc_weiz-tic_weiz)


        tic_eng = time.perf_counter()
        a_median_eng = eng.find_median(t_0, A_input, x_0)
        toc_eng = time.perf_counter()
        f_eng = eng.f(A_input, a_median_eng)
        medians_eng_vec.append(f_eng)
        eng_time_vec.append(toc_eng - tic_eng)


        f_mean = eng.f(A_input, x_0)
        medians_mean_vec.append(f_mean)
        #f_sed = eng.f(A_input, medians_sed['medians'][0][j * K+l])
        #medians_sed_vec.append(f_sed)
    #sed_time_vec = sedumi_times[j]
    # medians_sed_vec = medians_sed['medians'][0][j*K:j*K+5]
    #mean_sed = jap_can_dan(N, sed_time_vec, 'r*', 'SeDuMi', axs)
    mean_lmc = jap_can_dan(N, lmc_time_vec, 'bo', 'Light', axs)
    mean_eng = jap_can_dan(N, eng_time_vec, 'gs', 'Eng1', axs)
    mean_weisz = jap_can_dan(N, weisz_time_vec, 'cs', 'Weiszfeld', axs)

    mean_mean = np.array(medians_lmc_vec).sum(axis=0)/K
    mean_lmc_acc = jap_can_dan(N-2, np.array(medians_lmc_vec)/mean_mean, 'bo', 'Light', axs2)
    jap_can_dan(N + 2, np.array(medians_eng_vec)/mean_mean, 'gs', 'Eng1', axs2)
    jap_can_dan(N - 6, np.array(medians_mean_vec)/mean_mean, 'c^', 'Mean', axs2)
   # jap_can_dan(N + 6, np.array(medians_sed_vec)/mean_mean, 'r<', 'SeDuMi', axs2)
    jap_can_dan(N + 9, np.array(medians_weisz_vec) / mean_mean, 'y>', 'Weiszfeld', axs2)
    #means_sed.append(mean_sed)
    means_lmc.append(mean_lmc)
    means_eng.append(mean_eng)
    means_weisz.append(mean_weisz)
#axs.plot(n_vec, means_sed, 'r', linewidth=2)
axs.plot(n_vec, means_lmc, 'b', linewidth=2)
axs.plot(n_vec, means_eng, 'g', linewidth=2)
axs.plot(n_vec, means_weisz, 'c', linewidth=2)
axs.set_xlabel("N: number of data points")
axs.set_ylabel("Computation time [sec] ")

axs2.set_xlabel("N: number of data points")
axs2.set_ylabel("Sum of distances(normalized by lmc's result) ")

#axs2.plot(n_vec, means_eng, 'g', linewidth=2)
plt.close(1)
plt.show()



