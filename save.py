import numpy as np
import math
import random
import GPy

class PWorld:
    maxX = 30.0
    minX = 0.0
    maxY = 30.0
    minY = 0.0

    goalX = 1.0
    goalY = 1.0

    def inWorld(self,x,y):
        return x <= self.maxX and y <= self.maxY  and x >= self.minX and y >= self.minY

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
        (dx,dy) = math.cos(angle), math.sin(angle)
        newX, newY =  (x + dx, y + dy)
        if not self.inWorld(newX,newY):
            return x,y
        return self.squash(newX,newY)

    def rewardFunction(self,x,y):
        if self.inGoal(x,y):
            return 10000.0
        return 0.0

    def inGoal(self,x,y):
        return self.l2norm(x,y,self.goalX,self.goalY) <= 1.0

    def sample_n(self,n):
        return [0, math.pi/2, math.pi, math.pi*3/2]
        # return [(2.0*math.pi) * np.random.random_sample() for i in range(n)]

    def Qestimate(self,x,y,V_prev):
        Q = {}

        for a in self.sample_n(4):
            tot = 0
            for i in range(5):
                result = self.computeResult(x,y,a)
                resultState= np.array([result])
                tot+= self.rewardFunction(*result) + (0.9*V_prev.predict(resultState)[0][0][0] if not self.inGoal(*result) else 0)
            Q[a] = tot/float(5)
        return Q

    def valueiteration(self):
        in_gp = np.array([[0.0,0.0]])
        out_gp = np.array([[0.0]])
        V_prev = GPy.models.GPRegression(in_gp,out_gp,GPy.kern.Matern32(input_dim=2))
        for i in range(50):
            print "iteration: " + str(i)
            in_gp_temp = np.array([[self.goalX,self.goalY]])
            out_gp_temp = np.array([[10000.0]])
            # for i in range(20*20):
            for x in xrange(int(self.maxX)):
                for y in xrange(int(self.maxY)):
                # x = np.random.random_integers(0,self.maxX+1)
                # y = np.random.random_integers(0,self.maxY+1)
                    if not self.inWorld(x,y):
                        continue
                    Q = self.Qestimate(x,y,V_prev)

                    in_gp_temp = np.vstack((in_gp_temp, np.array([[x,y]])))
                    out_gp_temp = np.vstack((out_gp_temp, np.array([[Q[max(Q, key=Q.get)]]])))
            
            if in_gp.shape[0] > 200:
                in_gp, out_gp = in_gp_temp, out_gp_temp
            else:
                in_gp = np.vstack((in_gp,in_gp_temp))
                out_gp = np.vstack((out_gp,out_gp_temp))
            
            V_prev = GPy.models.GPRegression(in_gp,out_gp,GPy.kern.Matern32(input_dim=2))
            print in_gp.shape, out_gp.shape
            

        return V_prev



    def testPolicy(self,P,x,y):
        for i in range(200):
            print x,y
            x, y = self.computeAction(x,y,P[(x,y)])

    def testValueFunction(self, V,x,y):
        for i in range(100):
            print x,y
            Q = {}
            for a in self.sample_n(60):
                Q[a] = V.predict(np.array([self.computeResult(x,y,a)]))[0]
            a = max(Q, key=Q.get)
            x, y = self.computeResult(x,y,a)

    def getValue(self,V,x,y):
        print V.predict(np.array([[x,y]]))






        
p = PWorld()
valueFunc = p.valueiteration()

# print "play"
# p.testValueFunction(valueFunc,1.0,1.0)