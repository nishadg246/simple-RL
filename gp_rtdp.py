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
        # elif self.inObstacle(x,y):
        #     return -30.0
        return 0.0

    def inGoal(self,x,y):
        return self.l2norm(x,y,self.goalX,self.goalY) <= 1.0

    def sample_n(self,n):
        # return [0, math.pi/2, math.pi, math.pi*3/2]
        return [(2.0*math.pi) * np.random.random_sample() for i in range(n)]

    ATTEMPTS = 2
    SAMPLES = 5
    def Qestimate(self,x,y,V_prev):
        Q = {}
        for a in self.sample_n(self.SAMPLES):
            tot = 0
            for i in range(self.ATTEMPTS):
                result = self.computeResult(x,y,a)
                resultState= np.array([result])
                tot += self.rewardFunction(*result) + (0.9*V_prev.predict(resultState)[0][0][0] if not self.inGoal(*result) else 0)
            Q[a] = tot/float(self.ATTEMPTS)
        return Q

    def GPFromDict(self,d):
        X = np.reshape(np.array(d.keys()),(-1,len(d.keys()[0])))
        y = np.reshape(np.array(d.values()),(-1,1))
        gp = GPy.models.GPRegression(X,y,GPy.kern.Matern32(input_dim=2))
        return gp, X, y

    def GPFromDict2(self,d):
        X = np.reshape(np.array(d.keys()),(-1,1))
        y = np.reshape(np.array(d.values()),(-1,1))
        gp = GPy.models.GPRegression(X,y,GPy.kern.Matern32(input_dim=1))
        return gp, X, y

    ITERS = 50
    VPGS = []
    global VGP

    def RTDP(self):
        self.VPGS = []

        # Run Value Iteration
        # D={}
        # D[(self.goalX,self.goalY)] = 10000.0

        # VGP, X, y = self.GPFromDict(D)
        # S, _, _ = self.RRTStartToGoal(29.0,29.0)
        # S = list(S)
        # for i in range(200):
        #     print "iter " + str(i)
        #     for (x,y) in S:
        #             Q = self.Qestimate(x,y,VGP)
        #             D[(x,y)] = max(Q.values())
            
        #     VGP, X, y = self.GPFromDict(D)
        #     self.VPGS.append((VGP,D))
        #     print X.shape, y.shape
        
        # V upperbound
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
            # VGP, X, y = self.GPFromDict(D_temp)

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

    def RRTStartToGoal(self,x,y):

        S = set([(x,y)])
        parents = {}
        counter = 0
        while True:
            counter+=1
            newX, newY = np.random.uniform(self.minX, self.maxX), np.random.uniform(self.minY, self.maxY)
            if counter%100 == 0:
                newX,newY = self.goalX, self.goalY
            dists = map(lambda (a,b): ((a,b),self.l2norm(a,b,newX,newY)), list(S))
            elem = min(dists, key = lambda t: t[1])
            newX,newY = self.stepTowards(elem[0][0],elem[0][1],newX,newY)
            if not self.inWorld(newX,newY):
                continue
            S.add((newX,newY))
            parents[(newX,newY)] = elem[0]
            if self.inGoal(newX,newY):
                return S, parents, (newX,newY)


    def stepTowards(self,x,y,newX,newY):
        # if self.l2norm(x,y,newX,newY) < 1.0:
        #     return (newX,newY)
        # else:
        theta = math.atan2(newY-y,newX-x)
        return (x + math.cos(theta), y + math.sin(theta))

    def testPolicy(self,P,x,y):
        for i in range(200):
            print x,y
            x, y = self.computeAction(x,y,P[(x,y)])

    def testValueFunction(self, V,x,y,n):
        path = []
        reachedGoal = False
        for i in range(n):
            path.append((x,y))
            Q = {}
            for a in self.sample_n(100):
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