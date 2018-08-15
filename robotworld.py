from shapely.geometry import box
from shapely.geometry import Point, Polygon, LineString
import GPy
import numpy as np
import math
import quadrature

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
        [[a,b]] = np.random.multivariate_normal(np.array([0,0]), np.array([[0.05,0],[0,0.05]]), 1)
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
    SAMPLES = 100

    # def q_estimate(self,x,y,V_prev):
    #     Q = {}
    #     Var = {}
    #     for a in self.sample_n(self.SAMPLES):
    #         tot = 0.0
    #         for _ in range(self.ATTEMPTS):
    #             result,_ = self.transition(x,y,a)
    #             resultState= np.array([result])
    #             val =  (0.9*V_prev.predict(resultState)[0][0][0]) + self.reward(*result)
    #             tot += val
    #         Q[a] = tot / float(self.ATTEMPTS)
    #         Var[a] = 0.0
    #     return Q, Var
    # def q_estimate(self,x,y,V_prev):
    #     Q = {}
    #     Var = {}
    #     A, I, X, Y, Wi = quadrature.compute_prereq(self.RGP)
    #     A2, I2, X2, Y2, Wi2 = quadrature.compute_prereq(V_prev)
    #     for a in self.sample_n(self.SAMPLES):
    #         nx,ny = self.angle_delta(x,y,a)
    #         b = np.array([nx,ny])
    #         val1 = quadrature.integrate(self.RGP,b,0.05,A, I, X, Y, Wi)
    #         val2 = quadrature.integrate(V_prev,b,0.05, A2, I2, X2, Y2, Wi2)
    #         Q[a] = val1[0] + 0.9*val2[0]
    #         Var[a] = (val1[1], 0.9*val2[1])
    #     return Q,Var

    gps= {}
    A= {}
    def q_estimate(self,x,y,V_prev):
        A = {}
        actions = np.linspace(0, 2 * math.pi, self.SAMPLES)
        # def flambda(a):

        OPT_Rand1D(f, prior, iters, actions, initNum, numAdd, saveimgs=False)



        Xs = {}
        Ys = {}
        gps = {}
        for i, a in enumerate(actions):
            (xp, yp) = self.angle_delta(x, y, a)
            X = np.random.multivariate_normal([xp,yp], [[0.05,0],[0,0.05]], (30,))
            Y = X.copy()
            Y = np.apply_along_axis(lambda x: [V_prev.predict(np.array([x]))[0][0][0]], 1, Y)
            Xs[a] = X
            Ys[a] = Y
            gps[a] = GPy.models.GPRegression(X, Y, GPy.kern.src.rbf.RBF(input_dim=2))
            A[a] = quadrature.integrate(gps[a], np.array([xp,yp]), 0.05, *quadrature.compute_prereq(gps[a]))
            print a, A[a]

        def extend():
            maxa = max(A, key=lambda a: A[a][0] + 2 * A[a][1])
            nx,ny = self.angle_delta(x, y, maxa)
            X = Xs[maxa]
            X2 = np.random.multivariate_normal([nx,ny], [[0.05,0],[0,0.05]], (4,))
            Xs[maxa] = np.vstack((X, X2))
            Y = Xs[maxa].copy()
            Ys[maxa] = np.apply_along_axis(lambda x: [V_prev.predict(np.array([x]))[0][0][0]], 1, Y)
            gps[maxa] = GPy.models.GPRegression(Xs[maxa], Ys[maxa], GPy.kern.src.rbf.RBF(input_dim=1))
            A[maxa] = quadrature.integrate(gps[maxa], np.array([nx,ny]), 0.05, *quadrature.compute_prereq(gps[maxa]))

        import matplotlib.pyplot as plt
        for i in range(30):
            extend()
            self.gps=gps
            t,r,f = [],[],[]
            for a in A:
                t.append(a)
                r.append(A[a][0])
                f.append(A[a][1])
            
                plt.clf()
                plt.errorbar(t, r, yerr=f, fmt='o')
                plt.savefig("../imgs/rtdp-%03d.png" % i)


        maxa = max(A, key=lambda a: A[a][0])
        Q = {}
        Var = {}
        A, I, X, Y, Wi = quadrature.compute_prereq(self.RGP)
        A2, I2, X2, Y2, Wi2 = quadrature.compute_prereq(V_prev)

        nx, ny = self.angle_delta(x, y, maxa)
        b = np.array([nx, ny])
        val1 = quadrature.integrate(self.RGP, b, 0.05, A, I, X, Y, Wi)
        val2 = quadrature.integrate(V_prev, b, 0.05, A2, I2, X2, Y2, Wi2)
        Q[a] = val1[0] + 0.9 * val2[0]
        Var[a] = (val1[1], 0.9 * val2[1])
        return Q, Var


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

p = PWorld()
p.RTDP()