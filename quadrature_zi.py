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
    var = 0.0
    return mean, var

def OPT(f,sdim,prior):
    X = rand
    Y = np.apply_along_axis(f, 1, X)
    gp.fit(X,Y)
    for i in range(iters):
        actions = np.random.uniform(prior[0] - 10 * prior[1], prior[0] + 10 * prior[1], (10000, 1))
        maxa = max(actions, key=lambda a: integrate_dim(gp,sdim,a,prior[0],prior[1]))
        print "MAX"

        xs = np.random.uniform(prior[0] - 10 * prior[1], prior[0] + 10 * prior[1], (10000, 1))
        chosen = bq_acquisition(datum, prior[0], prior[1], xs, numAdd)


    return actionEstimate