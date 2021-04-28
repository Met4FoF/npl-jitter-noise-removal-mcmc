'''
Script for analysing a measurand signal subject to noise and jitter effects
------------------------------------------------------------------------------
Author         LRW, KJ, PH  - National Physical Laboratory
Project        EMPIR Met4FoF
------------------------------------------------------------------------------
Version        10/01/2020 - Creation
               25/02/2020 - Debugging
               05/03/2020 - Signal generation comparison with MATALB code
               20/03/2020 - Debgugging/combining with Bayesian work - see link
               31/03/2020 - Creating version of the code that is a class - for agentbased framework
               18/12/2020 - Agent version reviewed by Lizzie and updated accordingly 

'''


# Import Modules 
import numpy as np
import math
import MCMCMH.mcmcm_decayexp




# Decaying exponential
def DecayExpFunction(a,b,f,x):
    '''
    decaying exponential function evaluated at x with parameters a, b and f
    '''
    return a*np.exp(-b*x)*np.sin(2*np.pi*f*x)


# Decaying exponential - first derivative
def DecayExpFunction1der(a,b,f,x):
    '''
    first derviative of the decaying exponential function evaluated at x with parameters a, b and f
    '''
    return a*np.exp(-b*x)*((2*np.pi*f)*np.cos(2*np.pi*f*x) - b*np.sin(2*np.pi*f*x))

# Decaying exponential - second derivative
def DecayExpFunction2der(a,b,f,x):
    '''
    second derviative of the decaying exponential function evaluated at x with parameters a, b and f
    '''
    return a*np.exp(-b*x)*((-np.power((2*np.pi*f),2))*np.sin(2*np.pi*f*x)
     - ((4*b*np.pi*f)*np.cos(2*np.pi*f*x)) +  b*np.sin(2*np.pi*f*x))


class MCMCMH_NJ():
    '''
    Bayesian Noise and jitter reduction algorithm. MCMC used to determine the noise and jitter variances. 
    Noise and jitter variances are then used in an iterative algorithm to remove the noise and jitter from the signal 
    '''


    def __init__(self, fs, ydata, N, niter, tol, m0w, s0w, m0t, s0t, Mc, M0, Nc, Q):
        'Setting initial variables'

        # variables for AnalyseSignalN and NJAlgorithm
        self.fs       = fs
        self.ydata    = ydata
        #self.xdata = xdata
        self.N        = N
        self.niter    = niter
        self.tol      = tol
        self.m0w = m0w
        self.s0w = s0w
        self.m0t = m0t
        self.s0t = s0t
        self.Mc = Mc
        self.M0 = M0
        self.Nc = Nc
        self.Q = Q
        #outs = MCMCMH.mcmcm_decayexp.main(ydata, xdata, m0w, s0w, m0t, s0t, Mc, M0, Nc, Q)
        #self.jitterSD = outs[0]
        #self.noiseSD = outs[1]



    def AnalyseSignalN(self):
        '''
        Analyse signal to remove noise and jitter providing signal estimates with associated
        uncertainty. Uses normalised independent variable
        '''

        # Initialisation
        self.N = np.int64(self.N) # Converting window length integer to int64 format
        m = np.size(self.ydata) # Signal data length
        m = np.int64(m)# Converting signal length to int64 format

        # Setting initial variables
        n = (self.N-1)//2
        # Covariance matric for signal values
        Vmat = np.zeros((np.multiply(2, self.N)-1,np.multiply(2, self.N)-1))
        # Sensitivtity vecotrs
        cvec = np.zeros(n,self.N)
        # Sensitivtity matrix
        Cmat = np.zeros((self.N, np.multiply(2, self.N)-1))
        # Initial signal estimate
        yhat0 = np.full((m,1), np.nan)
        # Final signal estimate
        yhat = np.full((m,1),np.nan)
        # Uncertainty infromation for estimates
        vyhat = np.full((m, self.N), np.nan)
        # Consistency matrix
        R = np.full((m,1), np.nan)

        # Values of normalised independent variables
        datax = np.divide(np.arange(-n, n+1),self.fs)

        outs = MCMCMH.mcmcm_decayexp.main(self.ydata, datax, self.m0w, self.s0w, self.m0t, self.s0t, self.Mc, self.M0, self.Nc, self.Q)
        self.jitterSD = outs[0]
        self.noiseSD = outs[1]
        # Loop through indices L of window
        L = 0
        if np.size(self.ydata)>10:
        #for L in range(1, m-self.N+1):
        #while
            # Index k of indices L of windows
            k = L+n
            #print(k)
            # Extract data in window
            datay = self.ydata[L:L+self.N]
            #Inital polynomial approximation
            p = np.polyfit(datax, datay, 3)
            pval = np.polyval(p, datax)
            yhat0[k] = pval[n]


            # Applying algortithm to remove noise and jitter
            [yhat[k], ck, vark, R[k]] = MCMCMH_NJ.NJAlgorithm(self, datax, datay, p, pval)
            print(yhat[k])
            # First n windows, start building the covariance matrix Vmat for the data
            if L < n+1:
                Vmat[L-1, L-1]  = vark

            # for windows n+1 to 2n, continue building the covariance matrix Vmat and start stroing the
            # sensitivtity vectors ck in cvec
            elif L > n and L < np.multiply(2, n)+1:
                Vmat[L-1,L-1]   = vark
                cvec[L-n-1,:] = ck


            # For windows between 2n+1 and 4n, continue to build Vmat and cvec, and start building the sensitivtity
            # matrix Cmat from the sensitivtity vecotrs. Also, evaluate uncertainties for pervious estimates.
            elif L > np.multiply(2,n) and L < np.multiply(4,n)+2:

                Vmat[L-1,L-1] = vark
                # Count for building sensitivtity matrix
                iC = L-np.multiply(2,n)
                # Start building sensitivtity matrix from cvec
                Cmat[iC-1,:] = np.concatenate((np.zeros((1, iC-1)), cvec[0,:], np.zeros((1, self.N-iC))), axis=None)


                # Removing the first row of cvec and shift every row up one - creating
                # empty last row
                cvec = np.roll(cvec, -1, axis=0)
                cvec[-1,:] = 0
                # Update empty last row
                cvec[n-1,:] = ck
                Cmatk = Cmat[0:iC, 1:self.N-1+iC]

                # Continue building Vmat
                Vmatk = Vmat[1:self.N-1 + iC, 1:self.N-1 + iC]
                V = np.matmul(np.matmul(Cmatk,Vmatk), np.transpose(Cmatk))
                vhempty = np.empty((1, self.N-iC))
                vhempty[:] = np.nan
                # Begin building vyhat
                vyhat[L,:] = np.concatenate((vhempty, V[iC-1,:]),axis=None)


            # For the remaining windows, update Vmat, Cmat and cvec. Continue to
            # evaluate the uncertainties for previous estimates.
            elif L > np.multiply(4,n)+1:
                # Update Vmat
                Vmat = np.delete(Vmat, 0, axis=0)
                Vmat = np.delete(Vmat, 0, axis=1)
                Vmatzeros = np.zeros([Vmat.shape[0]+1, Vmat.shape[1]+1])
                Vmatzeros[:Vmat.shape[0], :Vmat.shape[1]] = Vmat
                Vmat = Vmatzeros
                Vmat[2*self.N-2, 2*self.N-2] = vark

                # Building updated Cmat matrix
                Cmat_old = np.concatenate((Cmat[1:self.N, 1:2*self.N-1], np.zeros([self.N-1,1])), axis=1)
                Cmat_new = np.concatenate((np.zeros([1,self.N-1]), cvec[0,:]), axis=None)
                Cmat = np.concatenate((Cmat_old, Cmat_new[:,None].T), axis=0)

                # Update cvec
                cvec = np.roll(cvec, -1, axis=0)
                cvec[-1,:] = 0

                # Continue building vyhat
                V = np.matmul(np.matmul(Cmat,Vmat), np.transpose(Cmat))
                vyhat[L,:] = V[self.N-2,:]

        L += 1

        return(yhat[k])




    def NJAlgorithm(self, datax, datay, p0, p0x):
        '''
        Noise and Jitter Removal Algorithm- Iterative scheme that preprocesses data to reduce the effects of 
        noise and jitter, resulting in an estimate of the true signal along with its associated uncertainty.
        
        Refer paper for details: https://ieeexplore.ieee.org/document/9138266 
        '''

        # Initialisatio
        iter_ = 0
        delta = np.multiply(2, self.tol)

        # Values of basis function at central point in window
        N = np.size(datax)
        n = np.divide(self.N-2, 2)
        k = np.int64(n+1)
        t = np.array([np.power(datax[k],3),np.power(datax[k],2),datax[k],1])
        # Deisgn Matrix
        X = np.array([ np.power(datax,3)+3*np.multiply(np.power(self.jitterSD,2),datax) ,
          np.power(datax,2)+np.power(self.jitterSD,2), datax,  np.ones(np.size(datax))  ])
        X = X.T

        # Iterative algortithm
        while delta>=self.tol:
            # Increment number of iterations
            iter_ = iter_ + 1

            # Step 2 - Polynomial fitting over window
            pd = np.polyder(p0)
            pdx = np.polyval(pd, datax)

            # Step 4
            # Weight calculation
            w = np.divide(1,[np.sqrt(np.power(self.jitterSD,2)*np.power(pdx,2)
             + np.power(self.noiseSD,2))])
            w = w.T

            # Calculating polynomial coeffs
            Xt = np.matmul(np.diagflat(w), X)
            C = np.matmul(np.linalg.pinv(Xt), np.diagflat(w))
            datay = datay.T
            p1 = np.matmul(C,datay)
            p1x = np.polyval(p1, datax)


            # Step 5 - stablise process
            delta = np.max(np.abs(p1x - p0x))
            p0 = p1
            p0x = p1x

            if iter_ == self.niter:
                print('Maximum number of iterations reached')
                break


        # Evaluate outputs
        c = np.matmul(t,C)
        yhat = np.matmul(c,datay)
        pd = np.polyder(p0)
        pdx = np.polyval(pd, datax[k])
        vk = np.power(self.jitterSD,2)*np.power(pdx,2) + np.power(self.noiseSD,2)
        R  = np.power(np.linalg.norm(np.matmul(np.diagflat(w),(datay - np.matmul(X,p0)))),2)


        return yhat, c, vk, R




def random_gaussian_whrand(M,mu,sigma,istate1,istate2):
    '''
    Generates random numbers from a Gaussian Distribution using the Wichmann–Hill random number generator 
    
    Inputs: 
    M:        Number of random numbers required 
    mu:       Mean of the Gaussian 
    sigma:    Std of the Gaussian 
    istate1:  vector of 4 integers  
    istate2:  vector of 4 different integers 
    
    Output: 
    xdist:    Random Gaussian vector of size M 
    '''
    mn = 0
    ndist = np.zeros(M)
    while mn < M:
        rr1 = whrand(istate1,1)
        rr2 = whrand(istate2,1)

        v1, istate1 = rr1
        v2, istate2 = rr2

        if mn < M:
            ndist[mn] = math.sqrt(-2*math.log(v1))*math.cos(2*math.pi*v2)
            mn = mn + 1

        if mn < M:
            ndist[mn] = math.sqrt(-2*math.log(v1))*math.sin(2*math.pi*v2)
            mn = mn + 1

        xdist = mu + sigma*ndist

    return xdist, istate1, istate2


def whrand(istate,N):
    '''
    Generates uniform random numbers between 0 and 1 using the Wichmann–Hill random number generator 
    
    Inputs: 
    istate:  vector of 4 integers  
    N:       Number of random numbers required  
    
    Outputs: 
    r:       Vector uniform random numbers of size M 
    istate:  Output vector of 4 integers 
    '''
    
  # Constants
    a = np.array([  11600, 47003, 23000, 33000 ])
    b = np.array([ 185127, 45688, 93368, 65075 ])
    c = np.array([  10379, 10479, 19423,  8123 ])
    d = np.array([ 456,   420,   300,  0 ]) + 2147483123

    r = np.zeros(N)
    for i in range(N):
        # Update states
        for j in range(4):
            istate[j] = a[j]*np.mod(istate[j], b[j]) - c[j]*np.fix(istate[j]/b[j])
            if istate[j] < 0:
                istate[j] = istate[j] + d[j]

        # Evaluate random number
        w = np.sum(np.divide(istate,d))
        r[i] = np.remainder(w,1)

    return r, istate


def testsignal():
    '''Wrapper file for noise and jitter removal. We generate a decaying exponential underlying signal
    with added noise and jitter.'''
    # Defining decaying exponential parameters
    a        = 10
    b        = 2
    f        = 2
    fs       = 100
    jitterSD = 0.003        #jitter
    noiseSD  = 0.002        #noise

    # Generating underlying signal
    x        = np.arange(0, 1, 1/fs)
    mm       = np.size(x)

    # Set parameters for noise and jitter removal
    N        = 15       #Window length
    niter    = 100      #Number of iterations
    tol      = 1e-9     #Tolerance

    # Setting parametrs for Monte Carlo loo
    M     = 10
    yy    = np.zeros((mm, M))
    yhat0 = np.zeros((mm,M))
    yhat  = np.zeros((mm,M))
    vyhat = np.zeros((mm,N,M))


    istate1time = np.array([234, 6757, 324, 54])
    istate2time = np.array([546, 2345, 7589, 583])

    istate1y = np.array([5, 587, 635, 6865])
    istate2y = np.array([9843, 8475, 76, 87639])





    # Monte Carlo loop
#    for r in range(0,M):
#        # Generating underlying signal with Guassian noise and jitter
#        xrnd = random_gaussian_whrand(mm, 0, 1, istate1time, istate2time)
#        xn = np.add(x,np.multiply(jitterSD,xrnd[0]))

#        s        = decayexpfunction(a,b,f,xn)

#        yrnd = random_gaussian_whrand(mm, 0, 1, istate1y, istate2y)

#        y        = s + np.multiply(noiseSD, yrnd[0])
#        yy[:,r]  = y

#        # Noise and jitter removal for normalised variables
#        [aa , bb , cc ,dd] =  MCMCMH_NJ(fs, y, jitterSD, noiseSD, N, iter_, tol)
#        yhat[:,r]          = aa[:,0]
#        vyhat[:,:,r]       = bb
#        R                  = cc
#        yhat0[:,r]         = dd[:,0]



