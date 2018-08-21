import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import scipy.integrate as integrate


class Datum(object):
    def __init__(self, X, Y, gp, mu, var, z):
        self.X = X
        self.Y = Y
        self.gp = gp
        self.z = z
        self.mu = mu
        self.var = var

def integrate_dim(gp, sdim,a,b, B):
    length_scale = gp.kernel_.length_scale
    A = length_scale ** 2 * np.diag(np.ones(sdim))
    I = np.identity(sdim)

    X = gp.X_train_
    n, dim = X.shape
    z = np.zeros((n, 1))
    determ = np.linalg.det(np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    abinv = np.linalg.inv(A + B)
    for i in range(n):
        x = X[i, :sdim]
        xa = X[i, sdim:]
        expon = np.exp(-0.5 * np.dot(np.dot((x - b), abinv), (x - b).T))
        factor = np.exp(-0.5 * np.sum((xa-a)**2 / (length_scale ** 2)))
        z[i, :] = factor * determ * expon
    mean = (z.T).dot(np.atleast_2d(gp.alpha_))
    var_determ = np.linalg.det(2 * np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    Kz = cho_solve((gp.L_, True), z)
    var = var_determ - np.dot(z.T, Kz)
    return mean[0][0], var[0][0], z

def bq_acquisition(datum,b,B,xs,a,z,num,sdim):
    length_scale = datum.gp.kernel_.length_scale
    A = length_scale ** 2 * np.diag(np.ones(sdim))
    I = np.identity(sdim)

    var_determ = np.linalg.det(2 * np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    determ = np.linalg.det(np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    abinv = np.linalg.inv(A + B)

    variances = np.zeros((xs.shape[0]))
    for i in range(xs.shape[0]):
        x = xs[i, :]
        xa = np.concatenate((xs[i, :],a))
        Xnew = np.vstack((datum.gp.X_train_, xa))
        K = datum.gp.kernel_(Xnew)
        K[np.diag_indices_from(K)] += datum.gp.alpha
        ztemp = determ * np.exp(-0.5 * np.dot(np.dot((x - b), abinv), (x - b).T))
        ztemp = np.vstack((z, ztemp))
        L = cholesky(K, lower=True)
        Kz = cho_solve((L, True), ztemp)
        variances[i] = var_determ - np.dot(ztemp.T, Kz)
    # print xs[variances.argsort()[:num]].shape
    # print variances.argsort()[:num], variances[variances.argsort()[:num]]
    sorted_vars = variances.argsort()
    l = []
    for j in sorted_vars:
        if len(l)>=num:
            break
        if np.abs(datum.X - xs[j]).min() > 0.05:
            l.append(j)
    return xs[l]

class Datum(object):
    def __init__(self, X, Y, gp):
        self.X = X
        self.Y = Y
        self.gp = gp

def OPT(f,iters,sdim,adim,prior):
    X = np.random.normal(0,3,(100,2))
    Y = np.apply_along_axis(f, 1, X)
    kernel = RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(X, Y)

    datum = Datum(X,Y,gp)
    D = {}
    for i in range(iters):
        print "iter %d" % i
        actions = np.random.uniform(-2,2, (1000, adim))
        UCB = lambda a: integrate_dim(gp, sdim, a, prior[0], prior[1])[0] + 1000.0 * integrate_dim(gp, sdim, a, prior[0], prior[1])[1]

        D[i] =  (actions, np.apply_along_axis(UCB, 1, actions))

        maxa = max(actions, key=UCB)
        _,_,z = integrate_dim(gp, sdim, maxa,prior[0], prior[1])
        xs = np.random.uniform(prior[0] - 5 * prior[1], prior[0] + 5 * prior[1], (1000, sdim))
        chosen = bq_acquisition(datum, prior[0], prior[1], xs, maxa, z, 10,sdim)

        X = datum.X
        Y = datum.Y
        chosen_columns = np.repeat(np.reshape(maxa,(1,-1)),chosen.shape[0],axis=0)
        newX = np.hstack((chosen, chosen_columns))
        print newX
        Xnew = np.vstack((X, newX))
        Yadd = newX.copy()
        Yadd = np.apply_along_axis(f, 1, Yadd)
        Ynew = np.vstack((Y, Yadd))

        kernel = RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gp.fit(X, Y)
        datum = Datum(X,Y,gp)

    return D