from shapely.geometry import box
from shapely.geometry import Point, Polygon, LineString
import GPy
import numpy as np
import math


class PWorld:
    maxX = 30.0
    minX = 0.0
    maxY = 30.0
    minY = 0.0
    bounds = box(minX, minY, maxX, maxY)

    goalX = 3.0
    goalY = 3.0

    startX = 28.0
    startY = 28.0
    obstacles = [box(5.0, 5.0, 10.0, 25.0),box(15.0, 10.0, 25.0, 15.0),box(20.0, 20.0, 25.0, 25.0)]

    holder = None

    def in_obstacle(self,x,y):
        for obs in self.obstacles:
            if obs.contains(Point(x, y)):
                return True
    def in_bounds(self,x,y):
        return self.bounds.contains(Point(x,y))
    def in_goal(self,x,y):
        return self.norm(x,y,self.goalX,self.goalY) <= 2.0
    def in_world(self,x,y):
        return self.in_bounds(x,y) and not self.in_obstacle(x,y)
    def norm(self,x,y,a,b):
        return np.linalg.norm(np.array([x,y]) - np.array([a,b]))
    def dist_to_poly(self,x,y,l):
        dist = {}
        for (i,obs) in enumerate(l):
            dist[i] = obs.boundary.distance(Point(x,y))
        return dist,l[i],i

    def transition(self,x,y,angle):
        (xp,yp) = self.angle_delta(x,y,angle)
        [[a,b]] = np.random.multivariate_normal(np.array([0,0]), np.array([[.05,0],[0,.05]]), 1)
        xp+=a
        yp+=b
        l = LineString([(x,y),(xp,yp)])
        for obs in self.obstacles:
            if l.intersects(obs):
                temp = l.intersection(obs)
                return temp.coords[0], True
        return (xp,yp),False

    def reward(self,nx,ny):
        if self.in_goal(nx,ny):
            return 10000.0
        distToEdge, _, _ = self.dist_to_poly(nx,ny,[self.bounds])
        if distToEdge <2.0:
            return (2.0 - distToEdge) * -2000.0
        distToObs, _, _ = self.dist_to_poly(nx,ny,self.obstacles)
        if distToObs <2.0:
            return (2.0 - distToObs) * -2000.0
        return 0.0

    def angle_delta(self,x,y,angle):
        (dx,dy) = math.cos(angle), math.sin(angle)
        return x + dx, y + dy

    def sample_n(self,n):
        return np.linspace(0, (2.0 * math.pi), n)

    ATTEMPTS = 5
    SAMPLES = 50

    # def q_estimate(self,x,y,V_prev):
    #     Q = {}
    #     for a in self.sample_n(self.SAMPLES):
    #         result,_ = self.transition(x,y,a)
    #         resultState= np.array([result])
    #         val =  (0.9*V_prev.predict(resultState)[0][0][0]) + self.reward(*result)
    #         Q[a] = val
    #     return Q

    # def q_estimate(self,x,y,V_prev):
    #     Q = {}
    #     for a in self.sample_n(self.SAMPLES):
    #         tot = 0.0
    #         for _ in range(self.ATTEMPTS):
    #             result,_ = self.transition(x,y,a)
    #             resultState= np.array([result])
    #             val =  (0.9*V_prev.predict(resultState)[0][0][0]) + self.reward(*result)
    #             tot += val
    #         Q[a] = tot / float(self.ATTEMPTS)
    #     return Q

    def q_estimate(self,x,y,V_prev):
        Q = {}
        for a in self.sample_n(self.SAMPLES):
            nx,ny = self.angle_delta(x,y,a)
            Q[a] = self.integrate(V_prev,nx,ny,0.05)[0]
        return Q

    # def Qestimate2(self,x,y,V_prev):
    #     Q = {}
    #     for a in self.sample_n(self.SAMPLES):
    #         # tot = 0
    #         # for i in range(self.ATTEMPTS):
    #         #     result = self.computeResult(x,y,a)
    #         #     resultState= np.array([result])
    #         #     tot +=  0.95*V_prev.predict(resultState)[0][0][0] 
    #         # Q[a] = tot/float(self.ATTEMPTS)
    #         nx,ny = self.computeDeterministicTransition(x,y,a)
    #         Q[a] = self.integrate(V_prev,nx,ny,0.1)[0]
    #         # print Q[a]
    #     return Q

    # def QestimateVal(self,x,y,V_prev):
    #     Q = {}
    #     for a in self.sample_n(self.SAMPLES):
    #         result,reward = self.transitionReward(x,y,a)
    #         resultState= np.array([result])
    #         val =  V_prev.predict(resultState)[0][0][0]
    #         Q[a] = val
    #     return Q

    def GPFromDict(self,d):
        # mf = GPy.core.Mapping(2,1)
        # def mean_func(x):
        #     return [10000.0 * 0.9**(self.l2norm(x[0][0],x[0][1],self.goalX,self.goalY))]
        # mf.f = mean_func
        # mf.update_gradients = lambda a,b: None
        X = np.reshape(np.array(d.keys()),(-1,len(d.keys()[0])))
        y = np.reshape(np.array(d.values()),(-1,1))
        gp = GPy.models.GPRegression(X,y,GPy.kern.src.rbf.RBF(input_dim=2))#, mean_function=mf)
        return gp, X, y

    VPGS = []
    Dicts = []
    ITERS = 70
    def RTDP(self):
        self.VPGS = []
        self.Dicts = []

        D = {}
        for x in range(int(self.maxX + 1)):
            for y in range(int(self.maxY + 1)):
                if self.in_obstacle(x,y):
                    D[(x,y)] = 0.0
                else: 
                    D[(x,y)] = 20000.0*0.9**self.norm(self.goalX,self.goalY,x,y)
        D[(self.goalX,self.goalY)] = 20000.0
        VGP, _, _ = self.GPFromDict(D)
        self.VPGS.append(VGP)

        def TrialRecurseWrapper(ax,ay,VGP):
            def TrialRecurse(x,y,VGP,hit=False):
                print (x,y)
                if self.in_goal(x,y) or hit:
                    return
                Q = self.q_estimate(x,y,VGP)
                maxa = max(Q, key=Q.get)
                # print maxa, Q[maxa]
                (sx,sy),hit = self.transition(x,y,maxa)
                TrialRecurse(sx,sy,VGP,hit)
                VGP, _,_ = self.GPFromDict(D)
                Q = self.q_estimate(x,y,VGP)
                maxa = max(Q, key=Q.get)
                print maxa, (x,y), Q[maxa]
                newx,newy = int(x),int(y)
                if self.in_obstacle(newx,newy) or hit:
                    D[(newx,newy)] = Q[maxa] *0.1
                else:
                    D[(newx,newy)] = Q[maxa]
            TrialRecurse(ax,ay,VGP)

        for i in range(self.ITERS):
            print "ITERATION %d" % (i)
            TrialRecurseWrapper(self.startX,self.startY,VGP)
            self.Dicts.append(D)
            # VGP, _,_ = self.GPFromDict(D)
            # try:
            #     VGP.optimize()
            # except:
            #     pass
            self.VPGS.append(VGP)

        return VGP


    def policyIter(self):
        D = {}
        for x in range(int(self.maxX + 1)):
            for y in range(int(self.maxY + 1)):
                if self.in_obstacle(x,y):
                    D[(x,y)] = 0.0
                else: 
                    D[(x,y)] = 20000.0*0.95**self.norm(self.goalX,self.goalY,x,y)
        D[(self.goalX,self.goalY)] = 40000.0
        VGP, _, _ = self.GPFromDict(D)
        self.VPGS.append(VGP)

        for i in range(self.ITERS):
            print "ITERATION %d" % (i)
            D = {}
            for x in range(int(self.maxX + 1)):
                print x
                for y in range(int(self.maxY + 1)):
                    if abs(x-3) + abs(y-3) > i:
                        continue
                    if self.in_obstacle(x,y):
                        D[(x,y)] = 0.0
                    Q = self.q_estimate(x,y,VGP)
                    maxa = max(Q, key=Q.get)
                    D[(x,y)] = Q[maxa]

            self.Dicts.append(D)
            VGP, _,_ = self.GPFromDict(D)
            try:
                VGP.optimize()
            except:
                pass
            self.VPGS.append(VGP)

    def testValueFunction(self,VGP,x,y,n):
        path = []
        reachedGoal = False
        for i in range(n):
            print x,y
            path.append((x,y))
            Q = self.q_estimate(x,y,VGP)
            maxa = max(Q, key=Q.get)
            (x, y),_ = self.transition(x,y,maxa)
            if not reachedGoal and self.in_goal(x,y):
                print "REACHED IN %d" % (i)
                reachedGoal=True
        return path
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

        modY = np.apply_along_axis(lambda x: [self.reward(x[0],x[1])], 1, X)
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
p.RTDP()