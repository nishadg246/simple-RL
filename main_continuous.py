import numpy as np
import numpy.random
import scipy.stats as ss
import math
import sklearn.gaussian_process as gp
import random

def expected_improvement(x, gaussian_process, evaluated_loss, maximize=True, n_params=1):
    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if maximize:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not maximize)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement

def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, maximize=False,
                               bounds=(0, 10), n_restarts=25):
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, maximize, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x

class PWorld:
    maxX = 30.0
    minX = 0.0
    maxY = 30.0
    minY = 0.0
    
    robotX = 1.0
    robotY = 1.0

    goalX = 29.0
    goalY = 29.0

    bounds=np.array([[0.0,30.0],[0.0,30.0]])

    actions = [(0,1),(0,-1),(-1,0), (1,0)]

    def inWorld(self,x,y):
        return x < self.maxX and y < self.maxY  and x > self.minX and y > self.minY

    def l2norm(self,x,y,a,b):
        return np.linalg.norm(np.array([x,y]) - np.array([a,b]))

    def getState(self):
        print self.robotX, self.robotY

    def sample(self,direction):
        return (direction[0] + np.random.normal(0, 0.2), direction[1] + np.random.normal(0, 0.2))

    def computeAction(self,x,y,direction):
        (dx,dy) = self.sample(direction)
        newX, newY =  (x + dx, y + dy)
        if self.inWorld(newX, newY):
            return (newX, newY)
        return (x,y)

    def takeAction(self,direction):
        self.robotX,self.robotY = self.computeAction(self.robotX,self.robotY,angle)

    def rewardFunction(self,x,y):
        if self.l2norm(x,y,self.goalX,self.goalY) <= 2.0:
            return 100.0
        return 0.0

    # def learnValueFunction(self):
    #     V = []
    #     kernel = gp.kernels.Matern()
    #     VGP = gp.GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=10,normalize_y=True)

    def play(self):
        while True:
            raw_input("step?")
            direction = random.randint(0, 4)
            self.takeAction(actions[direction])
            print self.getState()


    def valueiteration(self,iters):
        V = [0]*(iters+1)
        alpha, epsilon =1e-5,1e-7
        kernel = gp.kernels.Matern()

        V_prev = gp.GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=10,normalize_y=True)
        in_gp = np.array([[0.0,0.0]])
        out_gp = np.array([0])
        V_prev.fit(in_gp,out_gp)
        V[0] = (V_prev, in_gp, out_gp)

        for i in range(iters):
            V_temp = gp.GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=10,normalize_y=True)
            in_gp_temp = []
            out_gp_temp = []
            print "iteration: " + str(i)
            for x in xrange(int(self.maxX)):
                for y in xrange(int(self.maxY)):
                    Q = {}
                    for a in self.actions:
                        action = self.computeAction(x,y,a)
                        loc = np.array([action])
                        Q[a] = self.rewardFunction(*action) + V[i][0].predict(loc)[0]
                    in_gp_temp.append((x,y))
                    out_gp_temp.append(Q[max(Q, key=Q.get)])
            V_temp.fit(in_gp_temp,out_gp_temp)
            V[i+1] = (V_temp, in_gp_temp, out_gp_temp)

        return V

    def valueiteration2(self,iters):
        V = [0]*(iters+1)
        alpha, epsilon =1e-5,1e-7
        kernel = gp.kernels.Matern()

        V_prev = gp.GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=10,normalize_y=True)
        in_gp = np.array([[0.0,0.0]])
        out_gp = np.array([0])
        V_prev.fit(in_gp,out_gp)
        V[0] = (V_prev, in_gp, out_gp)

        for i in range(iters):
            V_temp = gp.GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=10,normalize_y=True)
            in_gp_temp = []
            out_gp_temp = []
            print "iteration: " + str(i)
            for n in range(200):
                model.fit(in_gp_temp, out_gp_temp)
                next_sample = sample_next_hyperparameter(expected_improvement, V_temp, out_gp_temp, maximize=True, bounds=self.bounds, n_restarts=100)
                # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
                if np.any(np.abs(next_sample - xp) <= epsilon):
                    next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

            for x in xrange(int(self.maxX)):
                for y in xrange(int(self.maxY)):
                    Q = {}
                    for a in self.actions:
                        action = self.computeAction(x,y,a)
                        loc = np.array([action])
                        Q[a] = self.rewardFunction(*action) + V[i][0].predict(loc)[0]
                    in_gp_temp.append((x,y))
                    out_gp_temp.append(Q[max(Q, key=Q.get)])
            V_temp.fit(in_gp_temp,out_gp_temp)
            V[i+1] = (V_temp, in_gp_temp, out_gp_temp)

        return V

    def testValueFunction(self,V,x,y):
        for i in range(400):
            print x,y
            Q = {}
            for a in self.actions:
                Q[a] = V.predict(np.array([self.computeAction(x,y,a)]))[0]
            a = max(Q, key=Q.get)
            x, y = self.computeAction(x,y,a)

    def testValueList(self,Vlist,x,y):
        for i in range(400):
            print x,y
            Q = {}
            for a in self.actions:
                Q[a] = sum(map( lambda V: V[0].predict(np.array([self.computeAction(x,y,a)]))[0], Vlist))
            a = max(Q, key=Q.get)
            x, y = self.computeAction(x,y,a)


    def testPolicy(self,P,x,y):
        for i in range(200):
            print x,y
            x, y = self.computeAction(x,y,P[(x,y)])


p = PWorld()
value = p.valueiteration(20)


