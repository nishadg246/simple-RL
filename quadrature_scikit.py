import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt


def computeMean(z, Wi, y):
    return np.dot(np.dot(z.T, Wi), y)[0][0]


def computeVariance(gp, z, Wi, A, B, I):
    w = gp.kern.variance[0]
    determ = np.linalg.det(2 * np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    return (w * determ - np.dot(np.dot(z.T, Wi), z))[0][0]


def computeZ(gp, X, i, A, B, b, I):
    x = X[i, :]
    w = gp.kern.variance[0]
    determ = np.linalg.det(np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    expon = np.exp(-0.5 * np.dot(np.dot((x - b), np.linalg.inv(A + B)), (x - b).T))
    return w * determ * expon


def compute_prereq(gp):
    dim = gp.X.shape[1]
    A = gp.kern.lengthscale[0] * np.diag(np.ones(dim))
    I = np.identity(dim)
    X = gp.X
    Y = gp.Y
    K = gp.kern.K(X)
    Ky = K.copy()
    Wi, LW, LWi, W_logdet = GPy.util.linalg.pdinv(Ky)
    return A,I,X,Y,Wi


def integrate(gp, b, B):
    n, dim = gp.X_train_.shape
    length_scale = gp.kernel_.length_scale
    A = length_scale** 2 * np.diag(np.ones(dim))
    I = np.identity(dim)

    X = gp.X_train_
    z = np.zeros((n, 1))
    for i in range(n):
        x = X[i, :]
        determ = np.linalg.det(np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
        expon = np.exp(-0.5 * np.dot(np.dot((x - b), np.linalg.inv(A + B)), (x - b).T))
        z[i, :] = determ * expon

    mean = (z.T).dot(np.atleast_2d(gp.alpha_))

    var_determ = np.linalg.det(2 * np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    Kz = cho_solve((gp.L_, True), z)
    var = var_determ - np.dot(z.T, Kz)
    return mean, var

class Datum(object):
    def __init__(self, X, Y, gp, mu, var):
        self.X = X
        self.Y = Y
        self.gp = gp
        self.mu = mu
        self.var = var

def init(func, prior, numStart, lscale=1.0):
    # X = np.atleast_2d(np.linspace(-100,100, numStart)).T
    X = np.random.normal(prior[0], prior[1]*5,(numStart,1))
    Y = X.copy()
    Y = np.apply_along_axis(func, 1, Y)
    kernel = RBF(lscale, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(X, Y)
    mu,var = integrate(gp, prior[0], prior[1])
    return Datum(X,Y,gp,mu[0][0],var[0][0])

def extend(func, prior, numAdd, datum,lscale=1.0):
    X = datum.X
    Y = datum.Y

    Xadd = np.random.normal(prior[0], prior[1]*5, size=(numAdd, 1))
    Xnew = np.vstack((X, Xadd))
    Yadd = Xadd.copy()
    Yadd = np.apply_along_axis(func, 1, Yadd)
    Ynew = np.vstack((Y, Yadd))

    kernel = RBF(lscale, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(X, Y)
    mu, var = integrate(gp, prior[0], prior[1])
    return Datum(Xnew, Ynew, gp, mu[0][0], var[0][0])

def OPT_Rand1D(f, prior, iters, saveimgs=False):
    actionEstimate = {}
    actions = np.linspace(-10, 10, 35)

    for a in actions:
        actionEstimate[a] = init(f(a),prior,10)

    for i in range(iters):
        maxa = max(actionEstimate, key=lambda a: actionEstimate[a].mu + 100.0 * actionEstimate[a].var)
        print maxa
        actionEstimate[maxa] = extend(f(maxa),prior, 10, actionEstimate[maxa])
        if saveimgs:
            x,y,e = [],[],[]
            for a in actions:
                x.append(a)
                y.append(actionEstimate[a].mu)
                e.append(100*actionEstimate[a].var)
            plt.clf()
            plt.errorbar(x, y, yerr=e, fmt='o')
            plt.savefig("./imgs/" + "{0:0=4d}".format(i) + ".png")
    return actionEstimate

