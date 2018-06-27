import numpy as np
import numpy.random
import scipy.stats as ss
import math
import sklearn.gaussian_process as gp
import random
class PWorld:
    maxX = 30.0
    minX = 0.0
    maxY = 30.0
    minY = 0.0
    
    robotX = 1.0
    robotY = 1.0

    goalX = 29.0
    goalY = 29.0

    actions = [(0,1),(0,-1),(-1,0), (1,0)]

    def inWorld(self,x,y):
        return x < self.maxX and y < self.maxY  and x > self.minX and y > self.minY

    def l2norm(self,x,y,a,b):
        return np.linalg.norm(np.array([x,y]) - np.array([a,b]))

    def getState(self):
        print self.robotX, self.robotY

    def sample(self,direction):
        return (direction[0] + np.random.normal(0, 1), direction[1] + np.random.normal(0, 1))

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

    def valueiteration(self):
        alpha, epsilon =1e-5,1e-7
        kernel = gp.kernels.Matern()
        V_prev = gp.GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=10,normalize_y=True)
        in_gp = np.array([[0.0,0.0]])
        out_gp = np.array([0])
        V_prev.fit(in_gp, out_gp)
        for i in range(40):
            in_gp_temp = []
            out_gp_temp = []
            print "iteration: " + str(i)
            for x in xrange(int(self.maxX)):
                for y in xrange(int(self.maxY)):
                    Q = {}
                    for a in self.actions:
                        action = self.computeAction(x,y,a)
                        i = np.array([action])
                        Q[a] = self.rewardFunction(*action) + V_prev.predict(i)[0]
                    in_gp_temp.append((x,y))
                    out_gp_temp.append(Q[max(Q, key=Q.get)])
            in_gp = in_gp_temp
            out_gp = out_gp_temp
            V_prev.fit(in_gp,out_gp)

        return V_prev

    def testValueFunction(self,V,x,y):
        for i in range(600):
            print x,y
            Q = {}
            for a in self.actions:
                Q[a] = V.predict(np.array([self.computeAction(x,y,a)]))[0]
            a = max(Q, key=Q.get)
            x, y = self.computeAction(x,y,a)


    def testPolicy(self,P,x,y):
        for i in range(200):
            print x,y
            x, y = self.computeAction(x,y,P[(x,y)])


p = PWorld()
value = p.valueiteration()

print "play"
p.testValueFunction(value,1,1)


