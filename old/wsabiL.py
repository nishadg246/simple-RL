import numpy as np
import numpy.matlib
import math

dim = 2
priorMu = np.array([[0,0]])
priorVar = np.array([[1,0],[0,1]])
kernelVar = np.array([[2,0],[0,2]])
lam = 1
alpha = 0.8
numSamples = 100
f = lambda x:0

bb          = np.zeros((1,dim));
BB          = np.reshape(priorVar.diagonal(),(-1,1)).T
VV          = np.reshape(kernelVar.diagonal(),(-1,1)).T
jitterNoise = 1e-6;     

hypLims     = 70*np.ones((1,dim+1)); 

mu              = np.zeros((numSamples-1,1));
logscaling      = np.zeros((numSamples-1,1));
Var             = np.zeros((numSamples-1,1));
clktime         = np.zeros((numSamples-1,1));
lHatD_0_tmp     = np.zeros((numSamples,1));
loglHatD_0_tmp  = np.zeros(lHatD_0_tmp.shape);
hyp = np.zeros((1,dim+1))


xx = np.zeros((numSamples,dim));
xx[-1,:] = bb+1e-6;
currNumSamples = 1;

def pdist2_squared_fast(A,B):
    return (np.sum(A**2,axis=1) + np.sum(B**2,axis=1).T - 2*np.matmul(A,B.T))

for t in range(numSamples-1):
    xxIter          = xx[numSamples-currNumSamples:,:]
    loglHatD_0_tmp[numSamples-currNumSamples,:]  = f(xxIter[0,:])
    logscaling[t]   = np.max(loglHatD_0_tmp[numSamples-currNumSamples:])

    lHatD_0_tmp[numSamples-currNumSamples:] = np.exp(loglHatD_0_tmp[numSamples-currNumSamples:]- logscaling[t]);
    print numSamples-currNumSamples
    aa = alpha * np.min(lHatD_0_tmp[numSamples-currNumSamples:])

    lHatD = np.sqrt(np.abs(lHatD_0_tmp[numSamples-currNumSamples:]- aa)*2)
     

    hyp[0:,0]      = np.log(lam)
    hyp[0:,1:] = np.log(VV)

    lam = np.exp(hyp[0,0])
    VV              = np.exp(hyp[0,1:])
    xxIterScaled    = xxIter * np.matlib.repmat(np.sqrt(1/VV),currNumSamples,1)
    dist2           = pdist2_squared_fast(xxIterScaled, xxIterScaled)
    Kxx = lam**2 * (1/(np.prod(2*math.pi*VV)**0.5)) * np.exp(-0.5*dist2)
    Kxx = Kxx + lam**2 * (1/(np.prod(2*math.pi*VV)**0.5))*jitterNoise*np.eye(*Kxx.shape)
    Kxx = Kxx/2 + Kxx.conj().T/2
    # print Kxx.shape
    invKxx = Kxx + np.eye(*Kxx.shape) # Come fix this


    ww              = np.matlib.matmul(invKxx,lHatD)
    Yvar            = (VV*VV + 2*VV*BB)/VV; 
    postProd        = np.matlib.matmul(ww,ww.conj().T)

    xx2sq           = xxIter * np.matlib.repmat(np.sqrt(1/Yvar),currNumSamples,1)
    bbch            = bb * np.sqrt(1/Yvar)
    xx2sqFin        = pdist2_squared_fast(xx2sq,bbch);
    xxIterScaled2   = xxIter * np.matlib.repmat(np.sqrt(BB/(VV*Yvar)),currNumSamples,1);

    dist4           = pdist2_squared_fast(xxIterScaled2,xxIterScaled2);
    YY              = lam**4 * (1 / np.prod(4*math.pi**2*((VV*VV + 2*VV*BB)),axis=1)**0.5) * np.exp(-0.5 * (pdist2_squared_fast(xx2sqFin,-xx2sqFin) + dist4)) * postProd

    mu[t] = (aa + 0.5*np.sum(YY))


    # strtSamp = 2*range(2,:).*rand(1,dim) - 50;

    # newX = cmaes( 'expectedVarL', strtSamp.T, [],opts, lam, VV, lHatD, xxIter, invKxx, jitterNoise, bb, BB)

    newX = np.array([np.random.normal(0, 1),np.random.normal(0, 1)]).T

    xx[numSamples-currNumSamples,:] = newX;
    currNumSamples = currNumSamples + 1
log_mu = np.log(mu) + logscaling
print lHatD_0_tmp