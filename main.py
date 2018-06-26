import numpy as np
import numpy.random
import scipy.stats as ss
import math
import sklearn.gaussian_process as gp
import random
class PWorld:
    maxX = 100.0
    minX = 0.0
    maxY = 100.0
    minY = 0.0
    
    robotX = 1.0
    robotY = 1.0

    goalX = 90.0
    goalY = 90.0

    def inWorld(self,x,y):
        return x < self.maxX and y < self.maxY  and x > self.minX and y > self.minY

    def l2norm(x,y,a,b):
        return np.linalg.norm(np.array([x,y]),np.array([a,b]))

    def getState(self):
        print self.robotX, self.robotY


    def sample2(self,angle):
        ### Sample from gaussian MM
        norm_params = np.array([[-20, 1],[20, 1]])
        n_components = norm_params.shape[0]
        # Weight of each component, in this case all of them are 1/3
        weights = np.ones(n_components, dtype=np.float64) / n_components
        # A stream of indices from which to choose the component
        mixture_idx = numpy.random.choice(n_components, size=1, replace=True, p=weights)
        # y is the mixture sample
        y = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                           dtype=np.float64)
        return angle + y[0]

    def sample(self,angle):
        return angle

    def computeAction(self,x,y,angle):
        actualAngle = self.sample(angle)
        print actualAngle
        newX, newY =  (x + math.cos(actualAngle), y + math.sin(actualAngle))
        if self.inWorld(newX, newY):
            return (newX, newY)
        return (x,y)

    def takeAction(self,angle):
        self.robotX,self.robotY = self.computeAction(self.robotX,self.robotY,angle)

    def rewardFunction(self,x,y):
        if self.l2norm(x,y,goalX,goalY) <= 2.0:
            return 100.0
        return -10.0

    # def learnValueFunction(self):
    #     V = []
    #     kernel = gp.kernels.Matern()
    #     VGP = gp.GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=10,normalize_y=True)

    def play(self):
        while True:
            raw_input("step?")
            angle = random.uniform(0,math.radians(180))
            self.takeAction(angle)
            print self.getState()


     def learnPolicy(self):
        kernel = gp.kernels.Matern()
	    model = gp.GaussianProcessRegressor(kernel=kernel,
	                                        alpha=alpha,
	                                        n_restarts_optimizer=10,
	                                        normalize_y=True)
 







        
p = PWorld()
p.play()