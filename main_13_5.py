import numpy as np
from numpy import linalg as LA


def f(A_input, x):
    f_sum = 0
    for i in range(0, len(A_input)):
        f_sum += LA.norm(x - A_input[i, :], ord=2, axis=None, keepdims=False)
    return (f_sum)


def t_i(f, i):
    return ((1 / (400 * f)) * ((1 + 1 / 600) ** (i - 1)))


def PowerMethod(A,k):
    x = np.random.normal(0, 1, size=(len(A_input), 1))
    y = np.dot(A**k, x)
    u = y / LA.norm(y, ord=2, axis=None, keepdims=False)
    return(u)



def g_t_i(x,A_input,i,t):
    return((1 + (t ** 2) * (LA.norm(x - A_input[i, :], ord=2, axis=None, keepdims=False)) ** 2) ** 0.5)


def f_t_x(x,A_input,i,t):
    f_t_sum = 0
    for i in range(0, len(A_input)):
        f_t_sum += g_t_i(x,A_input,i,t)-np.log(1+g_t_i(x,A_input,i,t))
    return f_t_sum



def ApproxMinEig(x,t,epsilon,A_input):
    s = (len(A_input), len(A_input))
    A_sum = np.zeros(s)
    hessian_f_t = np.zeros(s)
    for i in range(0, len(A_input)):
        g_t_i_x = g_t_i(x, A_input, i, t)
        x_minus_ai_outer = np.outer(x - A_input[i, :], x - A_input[i, :], out=None)
        A_sum += (t ** 4 * x_minus_ai_outer) / ((1 + g_t_i_x ** 2) * g_t_i_x)
        hessian_f_t += (t ** 2 / (1 + g_t_i_x)) * (
                    np.identity(len(A_input)) - (t ** 2 * x_minus_ai_outer) / (g_t_i_x * (1 + g_t_i_x)))
    k = np.floor(np.log(n / epsilon))
    u = PowerMethod(A_sum, k)
    Lambda = np.dot(np.dot(np.transpose(u), hessian_f_t), u)
    return Lambda, u


def w_t(x, A_input, t):
    w_sum = 0
    for i in range(0, len(A_input)):
        g_t_i_x = g_t_i(x, A_input, i, t)
        w_sum += 1/(1+g_t_i_x)
    return (w_sum)


def Grad_f_t(x, A_input, t):
    s = (len(A_input))
    grad_f_t = np.zeros(s)
    for i in range(0, len(A_input)):
        g_t_i_x = g_t_i(x, A_input, i, t)
        grad_f_t += (t ** 2 * (x - A_input[i, :])) / (1 + g_t_i_x)
    return grad_f_t


def MatNorm(Q, x):
    return (np.dot(np.dot(np.transpose(x), Q), x))**0.5


def LocalCenter(y, x_0, t, epsilon, A_input):
    lambda_calc, ni = ApproxMinEig(x_0, t, epsilon, A_input)
    wty = w_t(y, A_input, t)
    Q_matrix = t**2*np.dot(wty, np.identity(len(A_input)))-(t**2*wty-lambda_calc)*np.outer(ni, np.transpose(ni))
    local_x_0=y
    min_x=local_x_0
    for i in range(1, int(np.floor(64*np.log(1/epsilon)))):
        #NEEDS REVISION************************************************************************************************
        grad_f_t_x_prev=Grad_f_t(local_x_0, A_input, t)
        x_i = f(A_input, local_x_0) + np.inner(grad_f_t_x_prev, x - local_x_0)+4*MatNorm(Q_matrix, x-local_x_0)
       # local_x_0=x_i
    return x_i




A_input = np.array([[1, 0], [0, 1]])
t = 1
epsilon = 0.01
n = len(A_input)
Point_coordinate_sum = A_input.sum(axis=0)
x_0 = Point_coordinate_sum / len(A_input)
f_star_wave = f(A_input, x_0)
t1 = t_i(f_star_wave, 2)
epsilon_star_wave = epsilon/3
t_star_wave = 2*n/(epsilon_star_wave*f_star_wave)
epsilon_ni = ((epsilon_star_wave/(7*n))**2)/8
epsilon_c = (epsilon_ni/(36))**(3/2)


#def LineSearch(y, t, t_tag, u, epsilon, epsilon_star_wave, f_star_wave, n):
epsilon_O = (epsilon*epsilon_star_wave/(160*n**2))**2
l_line = -6*f_star_wave
u_line = 6*f_star_wave


#def OneDimMinimizer(l_line, u_line, epsilon, L_boundry):




x=np.array([1, 1])
y = np.array([1, 0])

lll = LocalCenter(y, x_0, t, epsilon, A_input)

print(lll)
