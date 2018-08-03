from shapely.geometry import box
from shapely.geometry import Point, Polygon, LineString
import GPy
import numpy as np
import math


class PWorld:
    maxX = 20.0
    minX = 0.0
    maxY = 20.0
    minY = 0.0
    bounds = box(minX, minY, maxX, maxY)

    goalX = 3.0
    goalY = 3.0

    startX = 19.0
    startY = 19.0
    obstacles = [box(5.0, 5.0, 15.0, 15.0)]
    supports = []
    resolutionX = 40
    resolutionY = 40
    RGP = None

    def __init__(self):
        R = {}
        for x in np.linspace(0, self.maxX+1, self.resolutionX):
            for y in np.linspace(0, self.maxY+1, self.resolutionY):
                self.supports.append((x,y))
        
        for x in np.linspace(0, self.maxX+1, 40):
            for y in np.linspace(0, self.maxY+1, 40):
                R[(x,y)] = self.reward(x,y)
        self.RGP, _, _ = self.GPFromDict(R)

    def in_obstacle(self,x,y):
        for obs in self.obstacles:
            if obs.contains(Point(x, y)):
                return True
    def in_bounds(self,x,y):
        return self.bounds.contains(Point(x,y))
    def in_goal(self,x,y):
        return self.norm(x,y,self.goalX,self.goalY) <= 1.0
    def in_world(self,x,y):
        return self.in_bounds(x,y) and not self.in_obstacle(x,y)
    def norm(self,x,y,a,b):
        return np.linalg.norm(np.array([x,y]) - np.array([a,b]))
    def dist_to_poly(self,x,y,l):
        dist = {}
        for (i,obs) in enumerate(l):
            dist[i] = obs.boundary.distance(Point(x,y))
        return dist,l[i],i
    def find_closest_support(self,nx,ny):
        dists = map(lambda x:(x,self.norm(nx,ny,x[0],x[1])),self.supports)
        supp = min(dists, key=lambda x:x[1])
        return supp[0]

    def transition(self,x,y,angle):
        (xp,yp) = self.angle_delta(x,y,angle)
        [[a,b]] = np.random.multivariate_normal(np.array([0,0]), np.array([[0.5,0],[0,0.5]]), 1)
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
            return 20000.0
        distToEdge, _, _ = self.dist_to_poly(nx,ny,[self.bounds])
        distToEdge = min(distToEdge.values())
        if distToEdge <2.0:
            return (2.0 - distToEdge) * -3000.0
        distToObs, _, _ = self.dist_to_poly(nx,ny,self.obstacles)
        distToObs = min(distToObs.values())
        if distToObs <2.0:
            return (2.0 - distToObs) * -3000.0
        return 0.0

    def angle_delta(self,x,y,angle):
        (dx,dy) = math.cos(angle), math.sin(angle)
        return x + dx, y + dy

    def sample_n(self,n):
        return np.linspace(0, (2.0 * math.pi), n)
        

    ATTEMPTS = 5
    SAMPLES = 180

    # def q_estimate(self,x,y,V_prev):
    #     Q = {}
    #     for a in self.sample_n(self.SAMPLES):
    #         result,_ = self.transition(x,y,a)
    #         resultState= np.array([result])
    #         val =  (0.9*V_prev.predict(resultState)[0][0][0]) + self.reward(*result)
    #         Q[a] = val
    #     return Q

    def q_estimate(self,x,y,V_prev):
        Q = {}
        Var = {}
        for a in self.sample_n(self.SAMPLES):
            tot = 0.0
            for _ in range(self.ATTEMPTS):
                result,_ = self.transition(x,y,a)
                resultState= np.array([result])
                val =  (0.9*V_prev.predict(resultState)[0][0][0]) + self.reward(*result)
                tot += val
            Q[a] = tot / float(self.ATTEMPTS)
            Var[a] = 0.0
        return Q, Var
    # def q_estimate(self,x,y,V_prev):
    #     Q = {}
    #     Var = {}
    #     for a in self.sample_n(self.SAMPLES):
    #         nx,ny = self.angle_delta(x,y,a)
    #         val1 = self.integrate(self.RGP,nx,ny,0.05)
    #         val2 = self.integrate(V_prev,nx,ny,0.05)
    #         Q[a] = val1[0] + 0.9*val2[0]
    #         Var[a] = (val1[1], 0.9*val2[1])

    #     return Q,Var

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
        for (x,y) in self.supports:
            if self.in_obstacle(x,y):
                D[(x,y)] = 0.0
            else: 
                D[(x,y)] = 20000.0*0.9**self.norm(self.goalX,self.goalY,x,y)
        D[(self.goalX,self.goalY)] = 20000.0
        VGP, _, _ = self.GPFromDict(D)
        # VGP.optimize()
        self.VPGS.append(VGP)

        def TrialRecurseWrapper(ax,ay,VGP):
            def TrialRecurse(x,y,VGP,hit=False):
                print (x,y)
                if self.in_goal(x,y) or hit:
                    return
                Q,_ = self.q_estimate(x,y,VGP)
                maxa = max(Q, key=Q.get)
                (sx,sy),hit = self.transition(x,y,maxa)

                TrialRecurse(sx,sy,VGP,hit)

                VGP, _,_ = self.GPFromDict(D)
                Q,V = self.q_estimate(x,y,VGP)
                maxa = max(Q, key=Q.get)
                newx,newy = self.find_closest_support(x,y)
                # if self.in_obstacle(newx,newy) or hit:
                #     D[(newx,newy)] = 0.0
                #     print maxa, (x,y), 0.0
                # else:
                D[(newx,newy)] = Q[maxa]
                print maxa, (x,y),(newx,newy), Q[maxa], V[maxa]
            TrialRecurse(ax,ay,VGP)

        for i in range(self.ITERS):
            print "ITERATION %d" % (i)
            TrialRecurseWrapper(self.startX,self.startY,VGP)
            self.Dicts.append(D)
            VGP, _,_ = self.GPFromDict(D)
            # try:
            #     VGP.optimize()
            # except:
            #     pass
            self.VPGS.append(VGP)

        return VGP

    def testValueFunction(self,VGP,x,y,n):
        path = []
        reachedGoal = False
        for i in range(n):
            print x,y
            path.append((x,y))
            Q,_ = self.q_estimate(x,y,VGP)
            maxa = max(Q, key=Q.get)
            (nx, ny),_ = self.transition(x,y,maxa)
            if not reachedGoal and self.in_goal(nx,ny):
                print "REACHED IN %d" % (i)
                reachedGoal=True
            x,y = nx,ny
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
        return w*determ*expon

    def integrate(self,gp,ix,iy,v):
        dim = gp.X.shape[1]
        A = gp.kern.lengthscale[0]*np.diag(np.ones(dim))
        Ainv = np.linalg.inv(A)
        B = np.diag(np.array([v,v]))
        b = np.array([[ix,iy]])
        I = np.identity(dim)
        X = gp.X
        Y = gp.Y

        # modY = np.apply_along_axis(lambda x: [self.reward(x[0],x[1])], 1, X)
        # Y = 0.9*Y + modY
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