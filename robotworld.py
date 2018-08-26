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
    resolutionX = 20
    resolutionY = 20
    RGP = None

    def __init__(self):
        R = {}
        for x in np.linspace(0, self.maxX+1, self.resolutionX):
            for y in np.linspace(0, self.maxY+1, self.resolutionY):
                self.supports.append((x,y))
        
        for x in np.linspace(0, self.maxX+1, 20):
            for y in np.linspace(0, self.maxY+1, 20):
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
            return 10.0
        distToEdge, _, _ = self.dist_to_poly(nx,ny,[self.bounds])
        distToEdge = min(distToEdge.values())
        if distToEdge <1.0:
            return (1.0 - distToEdge) * -1.0
        distToObs, _, _ = self.dist_to_poly(nx,ny,self.obstacles)
        distToObs = min(distToObs.values())
        if distToObs <1.0:
            return (1.0 - distToObs) * -1.0
        return -0.1

    def angle_delta(self,x,y,angle):
        (dx,dy) = math.cos(angle), math.sin(angle)
        return x + dx, y + dy

    def angle_delta_np(self,s,angle):
        (dx,dy) = math.cos(angle), math.sin(angle)
        temp = s.copy()
        temp[0]+=dx
        temp[1]+=dy
        return temp

    def sample_n(self,n):
        return np.linspace(0, (2.0 * math.pi), n)


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