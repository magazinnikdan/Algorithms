import numpy as np
import cvxpy as cp
from time import process_time


def OneDimMinimizer(l, u, eps, g, L):
    x_i = [l]
    y_l = [l]
    y_u = [u]
    for i in range(1, int(np.ceil(np.log(L * (u-l) / eps)/np.log(1.5)))):
        z_l_i = (2*y_l[i-1] + y_u[i-1])/3
        z_u_i = (y_l[i-1] + 2*y_u[i-1])/3
        if g(z_l_i) <= g(z_u_i):
            y_l.append(y_l[i-1])
            y_u.append(z_u_i)
            if g(z_l_i) <= g(x_i[i-1]):
                x_i.append(z_l_i)
            else:
                x_i.append(x_i[i-1])
        else:
            y_l.append(z_l_i)
            y_u.append(y_u[i-1])
            if g(z_u_i) <= g(x_i[i - 1]):
                x_i.append(z_u_i)
            else:
                x_i.append(x_i[i - 1])
    return x_i[-1]


def LineSearch(y, t_prime, mu, eps, eps_star, n, f_star, P, f_t, grad_t, hess_t, w_t, g_t):
    eps_O = (eps * eps_star / (160 * n ** 2))**2
    l, u = -6 * f_star, 6 * f_star

    q = lambda alpha: f_t(t_prime,LocalCenter(y=y+alpha*mu, t=t_prime, eps=eps_O, hess_t=hess_t, g_t=g_t, P=P,
                                              grad_t=grad_t, w_t=w_t))

    alpha_prime = OneDimMinimizer(l, u, eps_O, q, t_prime*P.shape[0])
    return LocalCenter(y=y+alpha_prime*mu, t=t_prime, eps=eps_O, hess_t=hess_t, g_t=g_t, P=P, grad_t=grad_t, w_t=w_t)


def solveConvex(f, y, constraints):
    x = cp.Variable(y.shape)
    loss = cp.Minimize(f(x))
    prob = cp.Problem(loss, constraints=[cp.norm(x - y) <= constraints])
    prob.solve(solver=cp.MOSEK)
    return x.value.flatten()


def LocalCenter(y, t, eps, hess_t, g_t, P, grad_t, w_t):
    Lambda, u = ApproxMinEig(x=y, t=t, eps=eps, n=P.shape[0], hess_t=hess_t, g_t=g_t, P=P)
    Q = t ** 2 * w_t(t, y) * np.eye(P.shape[1]) - (t ** 2 * w_t(t, y) - Lambda)*u[:, np.newaxis].dot(u[:, np.newaxis].T)
    X = []
    X.append(y)
    for i in range(int(np.ceil(64*np.log(1/eps)))):
        X.append(
            solveConvex(lambda x: np.sum(np.linalg.norm(P-X[i-1])) + cp.matmul(grad_t(t, X[i-1]), x-X[i-1])
                                  + 4 * cp.quad_form(x-X[i-1], Q), y, 1 / (49 * t)))
        if np.linalg.norm(X[i] - X[i-1]) == 0.0:
            break
    return X[-1]


def PowerMethod(A, k):
    x = np.random.randn(A.shape[0],)
    y = np.linalg.multi_dot([A for i in range(k)]).dot(x)
    return y / np.linalg.norm(y)


def ApproxMinEig(x, t, eps, n, hess_t, g_t, P):
    A = np.einsum('ij,ik->jk',
                  t**2 * np.multiply(x-P, 1 / (1 + g_t(t, x))[:, np.newaxis])*(1 / np.sqrt(g_t(t, x)[:, np.newaxis])),
                  t**2 * np.multiply(x-P, 1 / (1 + g_t(t, x))[:, np.newaxis])*(1 / np.sqrt(g_t(t, x)[:, np.newaxis])))
    u = PowerMethod(A, int(np.ceil(np.log(n / eps)/ 10)))
    Lambda = u.T.dot(hess_t(t, x)).dot(u)
    return Lambda, u


def accurateMedian(P, eps):
    x_0 = np.mean(P, axis=0)
    f_star = np.sum(np.linalg.norm(P - x_0, axis=1))
    eps_star = eps / 3.0
    t_star = 2 * P.shape[0] / (eps_star * f_star)
    k = np.floor(np.log(400 * t_star / f_star) / np.log(1 + 1/600) + 1)
    t_i = lambda i: 1.0 / (400 * f_star) * (1 + 1/600) ** (i-1)
    X = []
    eps_v = 1/8 * (eps_star / (7 * P.shape[0])) ** 2
    eps_c = (eps_v / 36) ** 1.5

    g_t_i = lambda t, i, x: np.sqrt(1 + t ** 2 * np.linalg.norm(x - P[i])**2)
    g_t = lambda t, x: np.sqrt(1 + t**2 * np.linalg.norm(x - P, axis=1)**2)
    hess_f_t_i = lambda t, i ,x: t ** 2 / \
                               (1 + g_t_i(t,i,x)) *(np.eye(P.shape[1]) - t ** 2 *
                                                    (x - P[i])[:, np.newaxis].dot((x - P[i])[:, np.newaxis].T)/
                                                    (g_t_i(t,i,x) * (1 + g_t_i(t,i,x))))
    f_t = lambda t, x: np.sum(np.sqrt(1 + t**2 * np.linalg.norm(P - x,axis=1) ** 2)
                             - np.log(1 + np.sqrt(1 + t**2 * np.linalg.norm(P - x,axis=1) ** 2)))
    grad_t = lambda t, x: np.sum(t**2 * np.multiply(x - P, 1+g_t(t, x)[:, np.newaxis]), axis=0)
    hess_t = lambda t, x: np.sum(t**2/(1 + grad_t(t, x))) * np.eye(P.shape[1]) - \
                         np.einsum('ij,ik->jk',
                                   t**2 * np.multiply(x-P, 1 / (1+g_t(t, x))[:, np.newaxis])*(1 / np.sqrt(
                                       g_t(t, x)[:, np.newaxis])),
                                   t**2 * np.multiply(x-P, 1 / (1+g_t(t, x))[:, np.newaxis])*(1 / np.sqrt(
                                       g_t(t, x)[:, np.newaxis])))
    w_t = lambda t, x: np.sum(1 / (1 + g_t(t, x)))

    X.append(LineSearch(y=x_0, t_prime=t_i(1), mu=np.zeros(x_0.shape), eps=eps_c, eps_star=eps_star,
                        n=P.shape[0], f_star=f_star, P=P, f_t=f_t, grad_t=grad_t, hess_t=hess_t, w_t=w_t, g_t=g_t))

    for i in range(1, int(k+1)):
        if i > 10:
            break
        print('Iteration ', i, ':')
        lambda_i, u_i = ApproxMinEig(x=X[i-1], t=t_i(i), eps=eps_v, n=P.shape[0], hess_t=hess_t, g_t=g_t, P=P)
        X.append(LineSearch(y=X[i-1], t_prime=t_i(i+1), mu=u_i, eps=eps_c, eps_star=eps_star, n=P.shape[0],
                            f_star=f_star, P=P, f_t=f_t, grad_t=grad_t, hess_t=hess_t, w_t=w_t, g_t=g_t))

    return X[-1]


if __name__ == '__main__':
    P = np.random.randn(3, 2)
    eps = 0.8
    t = process_time()
    x = accurateMedian(P, eps)
    elapsed_time = process_time() - t
    print("Michael cohen took {} seconds".format(elapsed_time))
    print('solution is {}'.format(x))
    y = cp.Variable(P.shape[1])
    loss = cp.Minimize(sum(map(lambda i: cp.norm(P[i]-y), range(P.shape[0]))))
    prob = cp.Problem(loss)
    prob.solve()
    cost = lambda X: np.sum(np.linalg.norm(P-X, axis=1))
    print('Optimal value using CVXPy = ', cost(y.value.flatten()))
    print('Optimal value using Michael cohen = ', cost(x))
    print('desired eps = {}, obtained eps = {}'.format(eps, cost(x) / cost(y.value.flatten()) - 1))
