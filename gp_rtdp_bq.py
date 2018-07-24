import numpy as np
import math
import random
import GPy
import sys
import numpy as np
import scipy.stats
import scipy.optimize as optim
from bayesian_quadrature import BQ
from gp import GaussianKernel
import math

# seed the numpy random generator, so we always get the same randomness
# np.random.seed(8706)
# sys.setrecursionlimit(10000)

class PWorld:
    maxX = 30.0
    minX = 0.0
    maxY = 30.0
    minY = 0.0

    goalX = 0.0
    goalY = 0.0

    obstacles = [[5.0,20.0,5.0,20.0]]

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
        newX, newY = self.computeUnsquashedResult(x,y,angle)
        if not self.inWorld(newX,newY):
            return x,y
        return self.squash(newX,newY)

    def computeUnsquashedResult(self,x,y,angle):
        # Add noise 
        # angle = angle + np.random.normal(0, 0.1)
        newX,newY =  self.computeDeterministicTransition(x,y,angle)
        dx,dy = np.random.multivariate_normal([0,0], [[0.1,0],[0,0.1]])
        newX, newY = newX + dx, newY + dy
        return newX,newY

    def computeDeterministicTransition(self,x,y,angle):
        (dx,dy) = math.cos(angle), math.sin(angle)
        return (x + dx, y + dy)
        

    def rewardFunction(self,x,y):
        if self.inGoal(x,y):
            return 10000.0
        return 0.0

    def inGoal(self,x,y):
        return self.l2norm(x,y,self.goalX,self.goalY) <= 1.0

    def sample_n(self,n):
        return np.linspace(0,(2.0*math.pi),n)

    ATTEMPTS = 5
    SAMPLES = 360
    def Qestimate(self,x,y,V_prev):
        Q = {}
        for a in self.sample_n(self.SAMPLES):
            # if not self.inWorld(*self.computeDeterministicTransition(x,y,a)):
            #     pass
            tot = 0
            for i in range(self.ATTEMPTS):
                result = self.computeResult(x,y,a)
                resultState= np.array([result])
                tot +=  (0.95*V_prev.predict(resultState)[0][0][0] if not self.inGoal(*result) else 0) + self.rewardFunction(*result) 
            Q[a] = tot/float(self.ATTEMPTS)
        return Q

    

    def GPFromDict(self,d):
        mf = GPy.core.Mapping(2,1)
        # def mean_func(x):
        #     return [10000.0 * 0.95**(self.l2norm(x[0][0],x[0][1],self.goalX,self.goalY))]
        # mf.f = mean_func
        # mf.update_gradients = lambda a,b: None
        X = np.reshape(np.array(d.keys()),(-1,len(d.keys()[0])))
        y = np.reshape(np.array(d.values()),(-1,1))
        gp = GPy.models.GPRegression(X,y,GPy.kern.src.rbf.RBF(input_dim=2))#, mean_function=mf)
        return gp, X, y

    def mergeLastN(self,dictlist,n):
        d = dictlist[-n:]
        t = d[0].items()
        for i in d[1:]:
            t+= i.items()
        return dict(t)

    ITERS = 20
    VPGS = []
    Dicts = []
    global VGP

    def RTDP(self):
        self.VPGS = []
        self.Dicts = []

        D_init = {}
        D_init[(self.goalX,self.goalY)] = 10000.0
        for x in xrange(int(self.maxX)):
                for y in xrange(int(self.maxY)):
                    D_init[(x,y)] = 10000.0 * 0.9**(self.l2norm(x,y,self.goalX,self.goalY))
        # VGP, X, y = self.GPFromDict(D_init)
        # self.VPGS.append(VGP)

        self.Dicts.append(D_init)
        VGP, _,_ = self.GPFromDict(self.mergeLastN(self.Dicts,len(self.Dicts)))
        # VGP.optimize()
        self.VPGS.append(VGP)
        
        def TrialRecurseWrapper(ax,ay,iteration,VGP):
            D_temp = {}
            def TrialRecurse(x,y,VGP):
                print (x,y)
                
                if self.inGoal(x,y):
                    return
                Q = self.Qestimate(x,y,VGP)
                maxa = max(Q, key=Q.get)
                
                (sx,sy) = self.computeResult(x,y,maxa)
                TrialRecurse(sx,sy,VGP)
                # VGP, _,_ = self.GPFromDict(self.mergeLastN(self.Dicts+[D_temp],5))
                Q = self.Qestimate(x,y,VGP)
                maxa = max(Q, key=Q.get)
                print maxa, (sx,sy), Q[maxa]
                D_temp[(sx,sy)] = Q[maxa]
                
            TrialRecurse(ax,ay,VGP)
            return D_temp

        for i in range(self.ITERS):
            print "ITERATION %d" % (i)
            D_temp = TrialRecurseWrapper(30.0,30.0,i,VGP)
            self.Dicts.append(D_temp)
            VGP, _,_ = self.GPFromDict(self.mergeLastN(self.Dicts,len(self.Dicts)))
            # VGP.optimize()
            self.VPGS.append(VGP)

        return VGP

    def testValueFunction(self,VGP,x,y,n):
        path = []
        reachedGoal = False
        for i in range(n):
            path.append((x,y))
            Q = self.Qestimate(x,y,VGP)
            maxa = max(Q, key=Q.get)
            x, y = self.computeResult(x,y,maxa)
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