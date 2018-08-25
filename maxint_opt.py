import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import GPy


def integrate_dim(gp, sdim, a, b, B):
    n, _ = gp.X.shape
    length_scale = gp.kern.lengthscale[0]
    A = length_scale ** 2 * np.diag(np.ones(sdim))
    I = np.identity(sdim)
    X = gp.X
    z = np.zeros((n, 1))
    determ = np.linalg.det(np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    abinv = np.linalg.inv(A + B)
    for i in range(n):
        x = X[i, :sdim]
        xa = X[i, sdim:]
        factor = np.exp(-0.5 * np.sum((xa - a) ** 2 / (length_scale ** 2)))
        expon = np.exp(-0.5 * np.dot(np.dot((x - b), abinv), (x - b).T))
        z[i, :] = factor * determ * expon
    mean = np.dot(z.T, gp.posterior.woodbury_vector)
    var_determ = np.linalg.det(2 * np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    var = var_determ - np.dot(z.T, gp.posterior.woodbury_inv.dot(z))
    return mean[0][0], var[0][0], z

def integrate(gp, b, B):
    n, dim = gp.X.shape
    length_scale = gp.kern.lengthscale[0]
    A = length_scale ** 2 * np.diag(np.ones(dim))
    I = np.identity(dim)
    X = gp.X
    z = np.zeros((n, 1))
    determ = np.linalg.det(np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    abinv = np.linalg.inv(A + B)
    for i in range(n):
        x = X[i, :]
        expon = np.exp(-0.5 * np.dot(np.dot((x - b), abinv), (x - b).T))
        z[i, :] = determ * expon
    mean = np.dot(z.T, gp.posterior.woodbury_vector)
    var_determ = np.linalg.det(2 * np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    var = var_determ - np.dot(z.T, gp.posterior.woodbury_inv.dot(z))
    return mean[0][0], var[0][0], z

def bq_acquisition(gp, sdim, a, b, B, xs, z):
    length_scale = gp.kern.lengthscale[0]
    A = length_scale ** 2 * np.diag(np.ones(sdim))
    I = np.identity(sdim)
    X = gp.X

    var_determ = np.linalg.det(2 * np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    determ = np.linalg.det(np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    abinv = np.linalg.inv(A + B)
    variances = np.zeros((xs.shape[0]))
    for i in range(xs.shape[0]):
        x = xs[i, :]
        xa = np.concatenate((xs[i, :], a))
        Xnew = np.vstack((gp.X, xa))
        K = gp.kern.K(Xnew)
        K[np.diag_indices_from(K)] += 1e-10
        ztemp = determ * np.exp(-0.5 * np.dot(np.dot((x - b), abinv), (x - b).T))
        ztemp = np.vstack((z, ztemp))
        L = cholesky(K, lower=True)
        Kz = cho_solve((L, True), ztemp)
        variances[i] = var_determ - np.dot(ztemp.T, Kz)
    sorted_vars = variances.argsort()
    return xs[sorted_vars[:1]]

def bq_acquisition2(gp, b, B, xs, z, num):
    n, dim = gp.X.shape
    length_scale = gp.kern.lengthscale[0]
    A = length_scale ** 2 * np.diag(np.ones(dim))
    I = np.identity(dim)

    var_determ = np.linalg.det(2 * np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    determ = np.linalg.det(np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
    abinv = np.linalg.inv(A + B)
    variances = np.zeros((xs.shape[0]))
    for i in range(xs.shape[0]):
        x = xs[i, :]
        Xnew = np.vstack((gp.X, x))
        K = gp.kern.K(Xnew)
        K[np.diag_indices_from(K)] += 1e-10
        ztemp = determ * np.exp(-0.5 * np.dot(np.dot((x - b), abinv), (x - b).T))
        ztemp = np.vstack((z, ztemp))
        L = cholesky(K, lower=True)
        Kz = cho_solve((L, True), ztemp)
        variances[i] = var_determ - np.dot(ztemp.T, Kz)
    sorted_vars = variances.argsort()
    return xs[sorted_vars[:num]]

def OPT(f,gp,b,B,sdim,adim,iters,abounds,sbounds,beta,lscale=1.0):
    gps = []
    for i in range(iters):
        print "iter %d" % i
        actions = np.random.uniform(abounds[0],abounds[1], (2000, adim))
        def UCB(gp):
            def f(a):
                val = integrate_dim(gp, sdim, a, b,B)
                return val[0] + beta*np.sqrt(val[1])
            return f
        acq_val = np.apply_along_axis(UCB(gp), 1, actions)
        index = acq_val.argmax()
        maxa = actions[index]
        print maxa
        _,_,z = integrate_dim(gp, sdim, maxa,b,B)
        xs = np.random.uniform(sbounds[0],sbounds[1], (1000, sdim))
        chosen = bq_acquisition(gp, sdim, maxa, b,B, xs,z)
        X = gp.X
        Y = gp.Y
        chosen_columns = np.repeat(np.reshape(maxa,(1,-1)),chosen.shape[0],axis=0)
        newX = np.hstack((chosen, chosen_columns))
        Xnew = np.vstack((X, newX))
        Yadd = newX.copy()
        Yadd = f(Yadd)
        print Yadd
        Ynew = np.vstack((Y, Yadd))
        gp = GPy.models.GPRegression(Xnew, Ynew, GPy.kern.src.rbf.RBF(input_dim=sdim+adim,lengthscale=lscale),noise_var=0.0)
        gps.append(gp)
    return gps
