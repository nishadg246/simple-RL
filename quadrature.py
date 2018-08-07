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
    return lambda x: np.sin(x)
def p(a):
    return np.array([a]), 10.0

def OPT_Rand1D(f,alow,ahigh,numa,slow,shigh,iters):
    A = {}
    actions = np.linspace(alow, ahigh, numa)
    Xs = {}
    Ys = {}
    gps = {}

    for a in actions:
        X = np.random.normal(p(a)[0],p(a)[1], size=(4, 1))
        Y = X.copy()
        Y = np.apply_along_axis(f(a), 1, Y)
        Xs[a] = X
        Ys[a] = Y
        gps[a] = GPy.models.GPRegression(X, Y, GPy.kern.src.rbf.RBF(input_dim=1))
        # gps[a].optimize()
        A[a] = integrate(gps[a], p(a)[0],p(a)[1], *compute_prereq(gps[a]))


    def extend(rand=False):
        maxa = max(A, key=lambda a: A[a][0] + 3 * A[a][1])
        # if rand:
        #     maxa = np.random.choice(A.keys())
        X = Xs[maxa]
        X2 = np.random.normal(p(a)[0],p(a)[1], size=(10, 1))
        Xs[maxa] = np.vstack((X, X2))
        Y = Xs[maxa].copy()
        Ys[maxa] = np.apply_along_axis(f(maxa), 1, Y)
        gps[maxa] = GPy.models.GPRegression(Xs[maxa], Ys[maxa], GPy.kern.src.rbf.RBF(input_dim=1))
        # gps[a].optimize()
        A[maxa] = integrate(gps[maxa],p(a)[0],p(a)[1], *compute_prereq(gps[maxa]))

    for i in range(iters):
        extend(i%5==0)

    return A,gps

A,gps = OPT_Rand1D(flambda,-10,10,10,-20,20,100)
x,y,e = [],[],[]
for a in A:
    x.append(a)
    y.append(A[a][0])
    e.append(A[a][1])
    print a, A[a]
import matplotlib.pyplot as plt
plt.clf()
plt.errorbar(x, y, yerr=e, fmt='o')
plt.show()