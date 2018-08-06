import numpy as np
import GPy


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