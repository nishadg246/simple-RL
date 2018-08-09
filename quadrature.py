import numpy as np
import GPy
import scipy

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


def integrate(gp, b, v, A,I,X,Y,Wi):
    dim = gp.X.shape[1]
    B = v * np.diag(np.ones(dim))
    b = b
    z = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        z[i, :] = computeZ(gp, X, i, A, B, b, I)
    return (computeMean(z, Wi, Y), computeVariance(gp, z, Wi, A, B, I))


def flambda(a):
    return lambda x: np.cos(a*x)
def prior(a):
    return 0.0, 1.0
def normal(s):
    return scipy.stats.norm(0, 1).pdf(s)

class Datum(object):
    def __init__(self, X, Y, gp, mu, var):
        self.X = X
        self.Y = Y
        self.gp = gp
        self.mu = mu
        self.var = var

def init(func, prior, numStart):
    X = np.linspace(-10,10, numStart)
    X = np.reshape(X,(-1,1))
    Y = X.copy()
    Y = np.apply_along_axis(func, 1, Y)
    gp = GPy.models.GPRegression(X, Y, GPy.kern.src.rbf.RBF(input_dim=1))
    mu,var = integrate(gp, prior[0], prior[1], *compute_prereq(gp))
    return Datum(X,Y,gp,mu,var)

def extend(func, prior, numAdd, datum):
    X = datum.X
    Y = datum.Y

    Xadd = np.random.normal(prior[0], prior[1], size=(numAdd, 1))
    Xnew = np.vstack((X, Xadd))
    Yadd = Xadd.copy()
    Yadd = np.apply_along_axis(func, 1, Yadd)
    Ynew = np.vstack((Y, Yadd))

    gp = GPy.models.GPRegression(Xnew, Ynew, GPy.kern.src.rbf.RBF(input_dim=1))
    mu, var = integrate(gp, prior[0], prior[1], *compute_prereq(gp))

    return Datum(X,Y,gp,mu,var)


def OPT_Rand1D(f,alow,ahigh,numa,iters):
    actionEstimate = {}
    actions = np.linspace(alow, ahigh, numa)

    for a in actions:
        actionEstimate[a] = init(f(a),5)

    for i in range(iters):
        maxa = max(actionEstimate, key=lambda a: A[a].mu + 3 * actionEstimate[a].var)
        actionEstimate[maxa] = extend(f(maxa),prior(maxa), 10, actionEstimate[maxa])
    return actionEstimate

# A,gps = OPT_Rand1D(flambda,-10,10,20,100)
# x,y,e = [],[],[]
# for a in A:
#     x.append(a)
#     y.append(A[a][0])
#     e.append(A[a][1])
#     print a, A[a]
# import matplotlib.pyplot as plt
# plt.clf()
# plt.errorbar(x, y, yerr=e, fmt='o')
# plt.show()

