import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import math
import winsound
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


def eu_dist(x1):
    # return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5
    return ((x1[0] ) ** 2 + (x1[1] ) ** 2) ** 0.5

def f(A_input, x):
    # Input: Sampled points' matrix nxd (A_input), point in R_d(x)
    # Output: Scalar sum of L2 norm of  x-a_i
    # page 3
    f_sum = 0
    n = len(A_input[:, 1])
    for i in range(n):
        # f_sum += LA.norm(x - A_input[i, :], ord=2, axis=None, keepdims=False)
        f_sum += eu_dist(x - A_input[i, :])
    return f_sum
    # return np.sum(LA.norm(A_input-x, axis=1))


def f_new(A_input, x):
    # Input: Sampled points' matrix nxd (A_input), point in R_d(x)
    # Output: Scalar sum of L2 norm of  x-a_i
    # page 3
    # f_sum = 0
    # n = len(A_input[:, 1])
    # for i in range(n):
    #     #f_sum += LA.norm(x - A_input[i, :], ord=2, axis=None, keepdims=False)
    #     f_sum += eu_dist(x - A_input[i, :])
    # return (f_sum)
    return np.sum(LA.norm(A_input-x, axis=1))


def t_i(f, i):
    # Input: Scalar value (f) and scalar index (i)
    # Output: Scalar path parameter t
    # defined in algorithm 1, page 8
    return (1 / (400 * f)) * ((1 + 1 / 600) ** (i - 1))


def PowerMethod(A,k):
    # Input: Matrix dxd (A), scalar parameter (k)
    # Output: epsilone-approximate top eigenvector[1xd] of the symmetric PSD matrix A
    # defined in algorithm 5, page 24
    x_pm = np.random.normal(0, 1, size=(len(A[1, :]), 1))
    # y = np.dot(A**k, x)
    y = np.dot(LA.matrix_power(A, int(k)), x_pm)
    # u = y / LA.norm(y, ord=2, axis=None, keepdims=False)
    u = y / eu_dist(y)
    u = [u[0][0], u[1][0]]
    return u


def g_t_i(x,A_input,i,t):
    # Input: Sampled points' matrix nxd (A_input), point in R_d(x), scalar index(i), scalar path parameter (t)
    # Output: Scalar value of the funcion g_i(x)
    # auxilary function , page 6,  2.3
    # return((1 + (t ** 2) * (LA.norm(x - A_input[i, :], ord=2, axis=None, keepdims=False)) ** 2) ** 0.5)
    return(1 + (t ** 2) * (eu_dist(x - A_input[i, :]) ** 2) ** 0.5)



def f_t_x(x,A_input,t):
    # Input: Sampled points' matrix (A_input), point in R_d(x), scalar path parameter (t)
    # Output: Scalar value of the sum of funcions f_t(x)
    # auxilary function , page 6,  2.3
    f_t_sum = 0
    n = len(A_input[:, 1])
    for i in range(0, n):
        f_t_sum += g_t_i(x, A_input, i, t)-np.log(1+g_t_i(x, A_input, i, t))
    return f_t_sum


def f_t_x_new(x,A_input,t):
    return np.sum(np.sqrt(1 + t**2 * np.linalg.norm(A_input - x, axis=1) ** 2) - np.log(1 + np.sqrt(1 + t**2 * \
        np.linalg.norm(A_input - x, axis=1) ** 2)))

##########################################################################################


# def power_iteration(A, num_simulations: int):
#     # Ideally choose a random vector
#     # To decrease the chance that our vector
#     # Is orthogonal to the eigenvector
#     b_k = np.random.rand(A.shape[1])
#     for _ in range(int(num_simulations)):
#         # calculate the matrix-by-vector product Ab
#         b_k1 = np.dot(A, b_k)
#
#         # calculate the norm
#         b_k1_norm = eu_dist(b_k1, 0)
#
#         # re normalize the vector
#         b_k = b_k1 / b_k1_norm
#
#     return b_k


def max_eig(M):
    w, v = LA.eig(M)
    arg = np.argmax(w)
    vec = v[arg]
    # return vec/LA.norm(vec, ord=2, axis=None, keepdims=False)
    return vec/eu_dist(vec)

def ApproxMinEig(x,t,epsilon,A_input):
    # Input: point in R_d (x), path parameter (t), target accuracy (epsilon) and sampled points' matrix nxd (A_input).
    # Output: Approximated minimal eigenvalue[scalar] and it's eigenvector of the Hessian of f_t(x) [1xd]
    # Algorithm 2, page 9
    tic_ApproxMinEig= time.perf_counter()
    n = len(A_input[:, 1])
    d = len(A_input[1, :])
    s = (d, d)
    A_sum = np.zeros(s)
    hessian_f_t = np.zeros(s)
    for i in range(0, n):
        g_t_i_x = g_t_i(x, A_input, i, t)
        x_minus_ai_outer = np.outer(np.transpose(x - A_input[i, :]), x - A_input[i, :], out=None)
        # t is very small so A_sum has VERY small members.
        A_sum = A_sum+(t ** 4 * x_minus_ai_outer) / ((1 + g_t_i_x ) ** 2* g_t_i_x)
        # hessian_f_t is defined on page 13 A1
        hessian_f_t = hessian_f_t+ (t ** 2 / (1 + g_t_i_x)) * (
                    np.identity(d) - (t ** 2 * x_minus_ai_outer) / (g_t_i_x * (1 + g_t_i_x)))
    # k = 20+np.floor(np.log(n / epsilon))
    # u = PowerMethod(A_sum, k)
    u = max_eig(A_sum)
    #u = power_iteration(A_sum, k)
    Lambda = float(np.dot(np.dot(np.transpose(u), hessian_f_t), u))
    toc = time.perf_counter()
    #print("ApproxMinEig", toc - tic_ApproxMinEig)
    return Lambda, u


def w_t(x, A_input, t):
    # Input: point in R_d (x), path parameter (t), sampled points' matrix nxd (A_input).
    # Output: scalar weight w_t for further calculations
    # auxilary function , page 6,  2.3
    w_sum = 0
    n = len(A_input[:, 1])
    for i in range(0, n):
        g_t_i_x = g_t_i(x, A_input, i, t)
        w_sum += 1/(1+g_t_i_x)
    return (w_sum)


def Grad_f_t(x, A_input, t):
    # Input: point in R_d (x), path parameter (t), sampled points' matrix nxd (A_input).
    # Output: Gradient vector of x [1xd]
    # auxilary function , page 13,  A1
    n = len(A_input[:, 1])
    d = len(A_input[1, :])
    s = (d)
    grad_f_t = np.zeros(s)
    for i in range(0, n):
        g_t_i_x = g_t_i(x, A_input, i, t)
        grad_f_t = grad_f_t+(t ** 2 * (x - A_input[i, :])) / (1 + g_t_i_x)
    return grad_f_t


def MatNorm(Q, x):
    # Input: A matrix dxd (Q), point in R_d (x)
    # Output: Matrix defined norm (x_t*Q*x)^0.5 [scalar]
    # auxilary function, page 5,  2.1
    return (np.dot(np.dot(np.transpose(x), Q), x))**0.5


def LocalCenter(y, t, epsilon, A_input):
    # Input: point in R_d (y), path parameter (t), target accuracy (epsilon) and sampled points' matrix nxd (A_input).
    # Output: ??? [1xd]
    #algorithm 3, page 10
    lambda_calc, ni = ApproxMinEig(y, t, epsilon, A_input)
    d = len(A_input[1, :])
    wty = w_t(y, A_input, t)
    Q_matrix = t**2*np.dot(wty, np.identity(d))-(t**2*wty-lambda_calc)*np.outer(ni, np.transpose(ni))
    local_x_0 = y
    # for i in range(1, int(np.floor(64*np.log(1/epsilon)))):
    for i in range(1, 15):
        grad_f_t_x_prev = Grad_f_t(local_x_0, A_input, t)# defined as a function above
        # x_i = argmin_calc(ni, t, lambda_calc, y, A_input, Q_matrix, local_x_0, grad_f_t_x_prev)
        x_i = argmin_calc_num(t, A_input, Q_matrix, local_x_0, grad_f_t_x_prev)
        local_x_0 = x_i
    return x_i

def argmin_calc_num(t, A_input, Q, x_prev, grad_f_t_x_prev):
    alpha_val = 1/(49*t)
    x = np.arange(x_prev[0]-alpha_val, x_prev[0]+alpha_val, alpha_val/2)
    y = np.arange(x_prev[1]-alpha_val, x_prev[1]+alpha_val, alpha_val/2)
    val_min = np.inf
    x_min = val_min
    y_min = val_min
    for i in range(len(y)):
        for j in range(len(x)):
            x_in = np.array([x[j], y[i]])
            # val =  f(A_input, x_prev) + np.inner(grad_f_t_x_prev, x_in - x_prev) + 4 * (MatNorm(Q, x_in - x_prev)) ** 2
            val = np.inner(grad_f_t_x_prev, x_in - x_prev) + 4 * (MatNorm(Q, x_in - x_prev)) ** 2
            if val < val_min:
                val_min = val
                x_min = x[j]
                y_min = y[i]
    #         plt.plot(x[j], y[i], 'y4')
    # plt.plot(x_min, y_min, 'y*')
    # plt.plot(A_input[:, 0], A_input[:, 1], 'ro')
    # plt.show()
    return np.array([x_min, y_min])


# def argmin_calc(v, t, lambda_eig , y, A_input, Q, z, grad_f_t_x_prev):
#     # Input: eigenvector 1xd (ni) , path parameter (t), scalar eigenvalue (lambda_eig),   point in R_d (y), sampled points' matrix nxd (A_input).
#     # a matrix dxd (Q), point in R_d (x), gradient vector of x 1xd (grad_f_t_x_prev)
#     # Output: a point in R_d which minimizes the function defined in algorithm 3
#     # algorithm 3 using lemma 32, page10
#     gamma=grad_f_t_x_prev/8
#     v_norm_val =LA.norm(v, ord=2, axis=None, keepdims=False)
#     # v_norm_val = eu_dist(v, [0, 0])
#
#     alpha_val = 1/(49*t)
#     beta_val = float(t**2*w_t(y, A_input, t)-lambda_eig)
#     c1_val = LA.norm(np.inner(Q, (z-y-np.inner(inv(Q), (gamma)))), ord=2, axis=None, keepdims=False)
#     # c1_val = eu_dist(np.inner(Q, (z - y - np.inner(inv(Q), (gamma)))), [0, 0])
#
#     c2_val = float(np.inner(np.transpose(v), np.inner(Q, (z - y - np.inner(inv(Q), gamma)))))
#     # polynome coefficients from tne result of analytical solution of lemma 32
#     p4 = alpha_val
#     p3 = -2*alpha_val*beta_val*v_norm_val**2
#     p2 = alpha_val*beta_val**2*v_norm_val**4-c1_val
#     p1 = 2*c1_val*beta_val*v_norm_val**2-2*c2_val
#     p0 = -c1_val*beta_val**2*v_norm_val**4+c2_val*beta_val*v_norm_val**2
#     coeff = [p4, p3, p2, p1, p0]
#     # solving and choosing minimal real lambda
#     val_min = math.inf
#     d = len(A_input[1, :])
#     for r in range(0, 4):
#         if np.roots(coeff)[r].imag == 0:
#             Lambda = np.roots(coeff)[r].real-t**2*w_t(y, A_input, t)
#             x_lambda = np.inner(LA.inv((Q+Lambda*np.identity(d))), (np.inner(Q, z)+Lambda*y-gamma))
#             val = f(A_input, z) + np.inner(grad_f_t_x_prev, x_lambda - z) + 4 * (MatNorm(Q, x_lambda - z))**2
#             if val < val_min:
#                 x_star = x_lambda
#                 val_min = val
#     return x_star


def q_minimizer(y, t, A_input, alpha, u, epsilon_O):
    # Input: Point in R_d (y) path parameter (t), sampled points' matrix nxd (A_input), scalar value (alpha),
    # bad direction 1xd (u),target accuracy (epsilon_0)
    # Output: Point in R_d  closer to the geometric median.
    #defined in Algorithm 4 on page 10
    local_center_alpha = LocalCenter(y + alpha * u, t, epsilon_O, A_input)
    y_q_min = y + alpha * u
    return f_t_x(local_center_alpha,A_input,t)


def OneDimMinimizer(A_input, y, t, u,  l_line, u_line, epsilon_0, L_boundry):
    # Input:Interval [l_line, u_line] in R and target additive error (epsilon_0), scalar >0 (L_boundry)
    # Output: Scalar which is supposed to refine the result of the LineSearch algorithm
    #uses oracle "q_minimizer" defined in Algorithm 4 on page 10, page 37
    x_l = [l_line]
    y_l = [l_line]
    y_u = [u_line]
    print(int(np.ceil(math.log(L_boundry * (u_line - l_line) / epsilon_0, 3 / 2)))-1)
    for i in range(1, int(np.ceil(math.log(L_boundry * (u_line - l_line) / epsilon_0, 3 / 2)))-1):
    # for i in range(1, int(25)):
        z_l_i = (2 * y_l[i-1] + y_u[i-1]) / 3
        z_u_i = (y_l[i-1] + 2 * y_u[i-1]) / 3
        q_z_l = q_minimizer(y, t, A_input, z_l_i, u, epsilon_0)
        q_z_u = q_minimizer(y, t, A_input, z_u_i, u, epsilon_0)
        q_x = q_minimizer(y, t, A_input, x_l[-1], u, epsilon_0)
        if q_z_l <= q_z_u:
            y_l.append(y_l[i-1])
            y_u.append(z_u_i)
            if q_z_l <= q_x:
                x_l.append(z_l_i)
        elif q_z_l > q_z_u:
            y_l.append(z_l_i)
            y_u.append(y_u[i-1])
            if q_z_u <= q_x:
                x_l.append(z_u_i)
    y_q_last = y + x_l[-1] * u
    plt.plot(y_q_last[0], y_q_last[1], 'mx')
    # x_l = np.array(x_l)
    # plt.plot(x_l[:,0], x_l[:,1], 'm4')

    return x_l[-1]

def LineSearch(A_input, y, t, t_tag, u, epsilon, epsilon_star_wave, f_star_wave, n,i):
    # Input:Point in R_d (y) path parameter (t),path parameter (t_tag), bad direction 1xd (u), target accuracy (epsilon)
    # accuracy (epsilon_star_wave), scalar value of f(x_o) (f_star_wave), scalar number of sampled points (n)
    # Output:  Point in R_d  closer to the geometric median.
    # page 10
    #epsilon_0 = (epsilon*epsilon_star_wave/(160*n**2))**2# a VERY small number, leads to long loops
    tic_LineSearch = time.perf_counter()

    epsilon_0 = epsilon/10
    l_line = -6*f_star_wave
    u_line = 6*f_star_wave
    alpha_tag = OneDimMinimizer(A_input, y, t, u, l_line, u_line, epsilon_0, t_tag*n)
    x_local = LocalCenter(y+alpha_tag*u, t_tag, epsilon_0, A_input)
    plt.plot(x_local[0], x_local[1], 'g*')
    plt.plot([y[0], y[0] + 0.5 * float(u[0])], [y[1], y[1] + 0.5 * float(u[1])])
    toc = time.perf_counter()
    #print("LineSearch", toc - tic_LineSearch)
    return x_local




@profile
def michael_median(A_input):

    # Here the main algorithm begins. we define epsilon and the sampled points
    # Input: Sampled points' matrix nxd (A_input), target accuracy (epsilon)
    # Output: Geometric median [1xd]

    print(A_input)
    num_iterations = 10
    epsilon = 0.5
    i = 1
    ti = 100
    print_flag = 0

    #definitions from the paper, page 8
    n = len(A_input[:, 1])
    Point_coordinate_sum = A_input.sum(axis=0)
    x_0 = Point_coordinate_sum / n
    f_star_wave = f(A_input, x_0)
    # t1 = t_i(f_star_wave, 2)
    epsilon_star_wave = epsilon/3
    # t_star_wave = 2*n/(epsilon_star_wave*f_star_wave)
    epsilon_ni = ((epsilon_star_wave/(7*n))**2)/8
    # epsilon_c = (epsilon_ni/(36))**(3/2)
    # use of while instead of for so no need to calculate k
    x_main = [x_0]
    #ti = t_i(f_star_wave, i)
    plt.plot(x_0[0], x_0[1], 'bs')
    plt.plot(A_input[:, 0], A_input[:, 1], 'ro')
    tic_main = time.perf_counter()
    x_main_i = LineSearch(A_input, x_0, ti, ti, np.transpose([0, 0]), epsilon, epsilon_star_wave, f_star_wave, n, i)
    plt.plot(x_main_i[0], x_main_i[1], 'r4')
    #plt.annotate(f"NP_LS {i}", (x_main_i[0] , x_main_i[1]))
    plt.plot(x_0[0], x_0[1], 'bs')
    # print("x_main_i", x_main_i)

    #while ti <= t_star_wave:
    while i <= num_iterations:
        lambda_i, u_i = ApproxMinEig(x_main_i, ti, epsilon_ni, A_input)
        print(u_i)

        x_main.append(x_main_i)
        i = i + 1
        #ti = t_i(f_star_wave, i)
        #ti_old = t_i(f_star_wave, i-1)
        ti_old = ti
        ti = ti*1.1
        x_main_i = LineSearch(A_input, x_main[-1], ti_old, ti, np.transpose([u_i[0], u_i[1]]), epsilon, epsilon_star_wave, f_star_wave, n,i)
        plt.plot(x_main_i[0], x_main_i[1], 'g3')
        plt.annotate(i-1, (x_main_i[0] , x_main_i[1]))
        print("x_main_i", x_main_i)
    toc = time.perf_counter()
    # print("Computation time",toc-tic_main)
    if print_flag:
        P_mesh=1
        x = np.arange(-P_mesh, P_mesh, 0.01)
        y = np.arange(-P_mesh, P_mesh, 0.01)
        xx, yy = np.meshgrid(x, y, sparse=True)
        s=[len(x),len(x)]
        z = np.zeros(s)
        z_min = np.inf
        x_min=z_min
        y_min = z_min
        for i in range(len(x)):
            for j in range(len(x)):
                x_in= [x[j],y[i]]
                z[i,j] = f_t_x(x_in, A_input, ti)
                # z[i,j] =  f(A_input, x_in)
                if z[i, j] < z_min:
                    z_min = z[i,j]
                    x_min = x[j]
                    y_min = y[i]
        h = plt.contourf(x, y, z)
        plt.plot(x_min, y_min, 'b2')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('scaled')
        plt.show()
        print("Error ",eu_dist(x_main_i-np.array([x_min, y_min])))
    return x_main_i

# if 1:
A_input = np.array([[1, 0], [-1, 0], [1, 0]])  #,[0, -1] ]) # A_input is a matrix containing the sampled points a_i
x = [1, 1]
t = 10
# UNCOMMENT FROM HERE
# #     A_input = np.array([[1, 0], [-1, 0], [0, 1]])
# # else:
# N = 10
# D = 2
# s = (N, D)
# A_input=np.zeros(s)
# for i in range(N):
#         A_input[i]=  np.random.normal(0, .5, D)
# # med = michael_median(A_input)
# # print(A_input)
# tic = time.perf_counter()
# print(f_t_x(x,A_input,t))
# t_old = time.perf_counter()-tic
#
# tic = time.perf_counter()
# print(f_t_x_new(x,A_input,t))
# t_new = time.perf_counter()-tic
# print(t_old/t_new)
# i = 1
# print("g_t_i(x,A_input,i,t)", g_t_i(x,A_input,i,t))
# print(((1 + (t ** 2) * (LA.norm(x - A_input[i, :], ord=2, axis=None, keepdims=False)) ** 2) ** 0.5))



