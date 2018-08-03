class Quadrature:
    @staticmethod
    def computeMean(z, Wi, y):
        return np.dot(np.dot(z.T, Wi), y)[0][0]

    @staticmethod
    def computeVariance(gp, z, Wi, A, B, I):
        lsq = gp.kern.lengthscale[0]
        w = gp.kern.variance[0]
        determ = np.linalg.det(2 * np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
        return w * determ - np.dot(np.dot(z.T, Wi), z)

    @staticmethod
    def computeZ(gp, X, i, A, B, b, I):
        x = X[i, :]
        lsq = gp.kern.lengthscale[0]
        w = gp.kern.variance[0]
        determ = np.linalg.det(np.dot(np.linalg.inv(A), B) + I) ** (-0.5)
        expon = np.exp(-0.5 * np.dot(np.dot((x - b), np.linalg.inv(A + B)), (x - b).T))
        return w * determ * expon

    @staticmethod
    def integrate(gp, b, v):
        dim = gp.X.shape[1]
        A = gp.kern.lengthscale[0] * np.diag(np.ones(dim))
        B = v * np.diag(np.ones(dim))
        b = b
        I = np.identity(dim)
        X = gp.X
        Y = gp.Y

        K = gp.kern.K(X)
        Ky = K.copy()
        Wi, LW, LWi, W_logdet = GPy.util.linalg.pdinv(Ky)

        z = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            z[i, :] = Quadrature.computeZ(gp, X, i, A, B, b, I)
        return (Quadrature.computeMean(z, Wi, Y), Quadrature.computeVariance(gp, z, Wi, A, B, I))