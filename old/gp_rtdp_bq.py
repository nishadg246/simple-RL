import random
import GPy
import numpy as np
import math
import polygons

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

    startX = 30.0
    startY = 30.0


    obstacles = [[(5.0, 5.0), (5.0, 25.0), (10.0, 25.0), (10.0, 5.0), (5.0, 5.0)],
                 [(15.0, 10.0), (15.0, 15.0), (25.0, 15.0), (25.0, 10.0), (15.0, 10.0)],
                 [(20.0, 20.0), (20.0, 25.0), (25.0, 25.0), (25.0, 20.0), (20.0, 20.0)]]


    context = polygons.new_context()
    for obs in obstacles:
        polygons.add_polygon(context, obs, [0, 1, 2, 3, 0])

    def inObstacle(self,x,y):
        return polygons.contains_points(self.context, [(x,y)])[0]

    def inBounds(self,x,y):
        return x < self.maxX and y < self.maxY  and x >= self.minX and y >= self.minY

    def inWorld(self,x,y):
        if self.inObstacle(x,y):
            return False
        return self.inBounds(x,y)

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
        # elif self.inObstacle(x,y):
        #     return -1000.0
        return 0.0

    def inGoal(self,x,y):
        return self.l2norm(x,y,self.goalX,self.goalY) <= 2.0

    # def inGoalStrict(self,x,y):
    #     return self.l2norm(x,y,self.goalX,self.goalY) <= 0.5

    def sample_n(self,n):
        return np.linspace(0,(2.0*math.pi),n)

    ATTEMPTS = 5
    SAMPLES = 40
    def Qestimate(self,x,y,V_prev):
        Q = {}
        for a in self.sample_n(self.SAMPLES):
            # tot = 0
            # for i in range(self.ATTEMPTS):
            #     result = self.computeResult(x,y,a)
            #     resultState= np.array([result])
            #     tot +=  0.95*V_prev.predict(resultState)[0][0][0] 
            # Q[a] = tot/float(self.ATTEMPTS)
            nx,ny = self.computeDeterministicTransition(x,y,a)
            Q[a] = self.integrate(V_prev,nx,ny,0.1)[0]
            # print Q[a]
        return Q

    # def Qestimate(self,x,y,V_prev):
    #     Q = {}
    #     for a in self.sample_n(self.SAMPLES):
    #         tot = 0
    #         for i in range(self.ATTEMPTS):
    #             result = self.computeResult(x,y,a)
    #             resultState= np.array([result])
    #             tot += (0.95*V_prev.predict(resultState)[0][0][0] if not self.inGoal(*result) else 0) + self.rewardFunction(*result) 
    #         Q[a] = tot/float(self.ATTEMPTS)
    #         # nx,ny = self.computeDeterministicTransition(x,y,a)
    #         # Q[a] = integrate(V_prev,nx,ny,0.1)
    #     return Q

    def GPFromDict(self,d):
        # mf = GPy.core.Mapping(2,1)
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

    ITERS = 40
    VPGS = []
    Dicts = []

    def RTDP(self):
        self.VPGS = []
        self.Dicts = []

        D_init = {}
        S,_,path = self.RRTStartToGoal(self.startX, self.startY)
        value = 10000.0
        for (x,y) in path:
            D_init[(x,y)] = value
            value*=0.9
        self.Dicts.append(D_init)
        VGP, _,_ = self.GPFromDict(self.mergeLastN(self.Dicts,1))
        self.VPGS.append(VGP)
        
        def TrialRecurseWrapper(ax,ay,iteration,VGP):
            D_temp = {}
            D_temp[(self.goalX,self.goalY)] = 10000.0
            def TrialRecurse(x,y,VGP):
                print (x,y)
                if self.inGoal(x,y):
                    return
                Q = self.Qestimate(x,y,VGP)
                maxa = max(Q, key=Q.get)
                (sx,sy) = self.computeResult(x,y,maxa)
                TrialRecurse(sx,sy,VGP)
                VGP, _,_ = self.GPFromDict(self.mergeLastN(self.Dicts+[D_temp],1))
                Q = self.Qestimate(x,y,VGP)
                maxa = max(Q, key=Q.get)
                print maxa, (x,y), Q[maxa]
                D_temp[(x,y)] = Q[maxa]
                
            TrialRecurse(ax,ay,VGP)
            return D_temp

        for i in range(self.ITERS):
            print "ITERATION %d" % (i)
            D_temp = TrialRecurseWrapper(self.startX,self.startY,i,VGP)
            self.Dicts.append(D_temp)
            VGP, _,_ = self.GPFromDict(self.mergeLastN(self.Dicts,1))
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

    def RRTStartToGoal(self,x,y):
        S = set([(x,y)])
        parents = {}
        counter = 0
        newX,newY = None,None
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
                break

        path = []
        path.append((newX,newY))
        while not newX==x and not newY ==y:
            newX,newY = parents[(newX,newY)]
            path.append((newX,newY))

        return S, parents, path

    def stepTowards(self,x,y,newX,newY):
        theta = math.atan2(newY-y,newX-x)
        return (x + math.cos(theta), y + math.sin(theta))

    def computeMean(self, z,Wi,y):
        return np.dot(np.dot(z.T,Wi),y)[0][0]

    def computeVariance(self,gp,z,Wi,A,B,I):
        lsq = gp.kern.lengthscale[0]
        w = gp.kern.variance[0]
        determ = np.linalg.det(2*np.dot(np.linalg.inv(A),B) + I)**(-0.5)
        return w*determ - np.dot(np.dot(z.T,Wi),z)

    def computeZ(self,gp,X,i,A,B,b,I):
        x = X[i,:]
        lsq = gp.kern.lengthscale[0]
        w = gp.kern.variance[0]
        determ = np.linalg.det(np.dot(np.linalg.inv(A),B) + I)**(-0.5)
        expon = np.exp(-0.5*np.dot(np.dot((x-b), np.linalg.inv(A+B)),(x-b).T))
    #     print expon.shape
        return w*determ*expon

    def integrate(self,gp,ix,iy,v):
        dim = 2
        A = gp.kern.lengthscale[0]*np.diag(np.ones(dim))
        Ainv = np.linalg.inv(A)
        B = np.diag(np.array([v,v]))
        b = np.array([[ix,iy]])
        I = np.identity(dim)
        X = gp.X
        Y = gp.Y

        modY = np.apply_along_axis(lambda x: [self.rewardFunction(x[0],x[1])], 1, X)
        Y = 0.9*Y + modY
        K = gp.kern.K(X)
        Ky = K.copy()
        GPy.util.diag.add(Ky, 1.0*1e-8)
        Wi, LW, LWi, W_logdet =  GPy.util.linalg.pdinv(Ky)

        z = np.zeros((X.shape[0],1))
        for i in range(X.shape[0]): 
            z[i,:] =self.computeZ(gp,X,i,A,B,b,I)

        return (self.computeMean(z,Wi,Y),self.computeVariance(gp,z,Wi,A,B,I))

     
p = PWorld()
valueFunc = p.RTDP()
# valueFunc = p.valueiteration()

# print "play"
# p.testValueFunction(valueFunc,1.0,1.0)