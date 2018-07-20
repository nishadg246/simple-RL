import numpy as np
import math
import random
import GPy
import sys
# sys.setrecursionlimit(10000)

class PWorld:
    maxX = 30.0
    minX = 0.0
    maxY = 30.0
    minY = 0.0

    goalX = 1.0
    goalY = 1.0

    obstacles = [[5.0,10.0,5.0,25.0],[15.0,25.0,10.0,15.0],[20.0,25.0,20.0,25.0]]
    # obstacles = []

    def inObstacle(self,x,y):
        for [x1,x2,y1,y2] in self.obstacles:
            if x>=x1 and x<=x2 and y>=y1 and y<=y2:
                return True
        return False

    def inWorld(self,x,y):
        if self.inObstacle(x,y):
            return False
        return x < self.maxX and y < self.maxY  and x >= self.minX and y >= self.minY

    def l2norm(self,x,y,a,b):
        return np.linalg.norm(np.array([x,y]) - np.array([a,b]))

    def squash(self,x,y):
        if self.inWorld(x,y):
            return (x,y)
        else:
            return (max(self.minX,min(self.maxX,x)),max(self.minY,min(self.maxY,y)))

    def computeResult(self,x,y,angle):
        # Add noise 
        angle = angle + np.random.normal(0, 0.5)
        newX,newY =  self.computeDeterministicTransition(x,y,angle)
        if not self.inWorld(newX,newY):
            return x,y
        return self.squash(newX,newY)

    def computeDeterministicTransition(self,x,y,angle):
        (dx,dy) = math.cos(angle), math.sin(angle)
        return (x + dx, y + dy)
        

    def rewardFunction(self,x,y):
        if self.inGoal(x,y):
            return 10000.0
<<<<<<< HEAD
        # elif self.inObstacle(x,y):
        #     return -30.0
=======
        elif self.inObstacle(x,y):
            return 0.0
>>>>>>> 55b5a664ccb5b8fd4674402900f40befda9bd251
        return 0.0

    def inGoal(self,x,y):
        return self.l2norm(x,y,self.goalX,self.goalY) <= 1.0

    def sample_n(self,n):
        # return [0, math.pi/2, math.pi, math.pi*3/2]
        return [(2.0*math.pi) * np.random.random_sample() for i in range(n)]

    ATTEMPTS = 5
    SAMPLES = 10
    def Qestimate(self,x,y,V_prev):
        Q = {}
        for a in self.sample_n(self.SAMPLES):
            tot = 0
            for i in range(self.ATTEMPTS):
                result = self.computeResult(x,y,a)
                resultState= np.array([result])
<<<<<<< HEAD
                tot += self.rewardFunction(*result) + (0.9*V_prev.predict(resultState)[0][0][0] if not self.inGoal(*result) else 0)
=======
                tot += self.rewardFunction(*result) + 0.9*(V_prev.predict(resultState)[0][0][0] if not self.inGoal(*result) else 0)
>>>>>>> 55b5a664ccb5b8fd4674402900f40befda9bd251
            Q[a] = tot/float(self.ATTEMPTS)
        return Q

    def GPFromDict(self,d):
        X = np.reshape(np.array(d.keys()),(-1,len(d.keys()[0])))
        y = np.reshape(np.array(d.values()),(-1,1))
        gp = GPy.models.GPRegression(X,y,GPy.kern.src.rbf.RBF(input_dim=2))
        return gp, X, y

    ITERS = 50
    VPGS = []

    def RTDP(self):
        self.VPGS = []

        D = {}
        for x in xrange(int(self.maxX)):
                for y in xrange(int(self.maxY)):
                    D[(x,y)] = 10000.0 * 0.9**(self.l2norm(x,y,self.goalX,self.goalY))
        VGP, X, y = self.GPFromDict(D)
        self.VPGS.append((VGP,D))

        D_temp = {}
        def MakeTrial(x,y):
            if self.inGoal(x,y):
                return
            Q = self.Qestimate(x,y,VGP)
            maxa = max(Q, key=Q.get)
            (sx,sy) = self.computeResult(x,y,maxa)
            MakeTrial(sx,sy)
            maxa = max(Q, key=Q.get)
            D_temp[(sx,sy)] = Q[maxa]

        for i in range(self.ITERS):
            # if len(D_temp) > 2000:
            #     D_temp = {}
            try:
                MakeTrial(29.0,29.0)
            except RuntimeError as re:
                if re.args[0] != 'maximum recursion depth exceeded in cmp':
                    # different type of runtime error
                    raise
                print "fail"
                continue
            VGP, X, y = self.GPFromDict(D_temp)
            self.VPGS.append((VGP,D_temp))
            print X.shape, y.shape
        return VGP

    def testValueFunction(self, V,x,y,n):
        path = []
        reachedGoal = False
        for i in range(n):
            path.append((x,y))
            Q = {}
<<<<<<< HEAD
            for a in self.sample_n(100):
=======
            for a in self.sample_n(150):
>>>>>>> 55b5a664ccb5b8fd4674402900f40befda9bd251
                Q[a] = V.predict(np.array([self.computeResult(x,y,a)]))[0]
            a = max(Q, key=Q.get)
            x, y = self.computeResult(x,y,a)
            if not reachedGoal and self.inGoal(x,y):
                print i
                reachedGoal=True
        return path

    def getValue(self,V,x,y):
        print V.predict(np.array([[x,y]]))






        
p = PWorld()
valueFunc = p.RTDP()

# print "play"
# p.testValueFunction(valueFunc,1.0,1.0)