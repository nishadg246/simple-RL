import numpy as np
import numpy.random
import scipy.stats as ss
import math
import sklearn.gaussian_process as gp
import random
class PWorld:
    maxX = 100
    minX = 0
    maxY = 100
    minY = 0
    
    robotX = 1
    robotY = 1

    goalX = 99
    goalY = 99

    actions = [(0,1),(0,-1),(-1,0), (1,0)]

    def inWorld(self,x,y):
        return x < self.maxX and y < self.maxY  and x > self.minX and y > self.minY

    def l2norm(self,x,y,a,b):
        return np.linalg.norm(np.array([x,y]) - np.array([a,b]))

    def getState(self):
        print self.robotX, self.robotY


    def sample2(self,angle):
        ### Sample from gaussian MM
        norm_params = np.array([[-20, 1],[20, 1]])
        n_components = norm_params.shape[0]
        # Weight of each component, in this case all of them are 1/3
        weights = np.ones(n_components, dtype=np.float64) / n_components
        # A stream of indices from which to choose the component
        mixture_idx = numpy.random.choice(n_components, size=1, replace=True, p=weights)
        # y is the mixture sample
        y = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                           dtype=np.float64)
        return angle + y[0]

    def sample(self,direction):
        return self.actions[direction]

    def computeAction(self,x,y,direction):
        (dx,dy) = direction
        newX, newY =  (x + dx, y + dy)
        if self.inWorld(newX, newY):
            return (newX, newY)
        return (x,y)

    def takeAction(self,direction):
        self.robotX,self.robotY = self.computeAction(self.robotX,self.robotY,angle)

    def rewardFunction(self,x,y):
        if self.l2norm(x,y,self.goalX,self.goalY) <= 0.0:
            return 100.0
        return -10.0

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

    def valueiteration(self):
        V = {}
        for i in range(200):
            print "iteration: " + str(i)
            Q = {}
            V_temp = {}
            for x in xrange(self.maxX):
                for y in xrange(self.maxY):
                    Q = {}
                    for a in self.actions:
                         Q[a] = self.rewardFunction(*self.computeAction(x,y,a)) + V.get(self.computeAction(x,y,a),0)
                    V[(x,y)] = Q[max(Q, key=Q.get)]
            # V =    V_temp
            for i in V:
                print i, V[i]

        print "Policy"
        P = {}        
        for x in xrange(self.maxX):
                for y in xrange(self.maxY):
                    Q = {}
                    for a in self.actions:
                         Q[a] = self.rewardFunction(*self.computeAction(x,y,a)) + V.get(self.computeAction(x,y,a),0)
                    P[(x,y)] = max(Q, key=Q.get)
        for i in P:
                print i, P[i]

        return P

    def testPolicy(self,P,x,y):
        for i in range(200):
            print x,y
            x, y = self.computeAction(x,y,P[(x,y)])


p = PWorld()
policy = p.valueiteration()

print "play"
p.testPolicy(policy,1,1)