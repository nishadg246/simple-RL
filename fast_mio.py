import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import GPy as gpy
import GPy
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import Axes3D
import maxint_opt as opt

DIM = 3
LSCALE = 1.0
SDIM = 2
ADIM = 1
BETA = 2.0
ABOUNDS = [-5.0,5.0]
SBOUNDS = [-5.0,5.0]

def sample_function(x_range, N=100, seed=4):
#     np.random.seed(seed)
    dx = x_range.shape[1]
    k = gpy.kern.src.rbf.RBF(input_dim=dx,lengthscale=LSCALE)
    x = np.random.uniform(x_range[0], x_range[1], (N, dx))
    cov = k.K(x, x)
    mu = np.zeros(x.shape[0])
#     y = np.atleast_2d(normal(x[:,1])).T
    y = np.random.multivariate_normal(np.squeeze(mu), cov)[:, None]
    m = gpy.models.GPRegression(x, y, k)#, mean_function=mean)
    m.likelihood.variance = 0.0

    def f(x): return m.predict(x)[0]
    return f,m,x


data = []

for i in range(5):
    print i
    x_range = np.array([[-5.0,-5.0,-5.0], [5.0,5.0,5.0]])
    f,m,x = sample_function(x_range)
    x = np.random.uniform(x_range[0], x_range[1], (1, DIM))
    y = f(x)
    gp = GPy.models.GPRegression(x, y, GPy.kern.src.rbf.RBF(input_dim=DIM,lengthscale=LSCALE),noise_var=0.0)
    b = np.zeros(SDIM)
    B = np.identity(SDIM)*0.5
    gps = opt.OPT(f,gp,b,B,SDIM,ADIM,80,ABOUNDS,SBOUNDS,BETA,LSCALE)

    actions = np.linspace(-6.0,6.0,1000)
    mus = np.array([])
    intervals = [10,20,30,40,50,60,70,79]
    maxs = []
    for a in actions:
        mus = np.append(mus,opt.integrate_dim(m,SDIM, a, b,B)[0])
    for interv in intervals:
        preds = np.array([])
        for a in actions:
            preds = np.append(preds,opt.integrate_dim(gps[interv],SDIM, a, b,B)[0])
        maxs.append((actions[preds.argmax()], preds.max()))
    data.append((actions[mus.argmax()], mus.max(),maxs))
    print data







