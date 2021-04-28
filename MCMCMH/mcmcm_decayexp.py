################################################################################################################
# MCMC estimation of noise and jitter variances
# KJ 14-08-2020
# Based on code on T drive FoF folder that has SQM stuff sorted and has been reviewed by Lizzie
#################################################################################################################



import numpy as np
import random as rnd
import math
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


"""### mcmcci: MCMC convergence indices for multiple chains."""

def mcmcci(A,M0):
    '''
    mcmcci: MCMC convergence indices for multiple chains.
    ----------------------------------------------------------------------------
    KJ, LRW, PMH

    Version 2020-04-22

    ----------------------------------------------------------------------------
    Inputs:
    A(M, N): Chain samples, N chains of length M.

    M0: Length of burn-in, M > M0 >= 0.

    Outputs:
    Rhat: Convergence index. Rhat is expected to be greater than 1. The closer
    Rhat is to 1, the better the convergence.

    Neff: Estimate of the effective number of independent draws. Neff is
    expected to be less than (M-M0)*N.
    ----------------------------------------------------------------------------
    Note: If the calculated value of Rhat is < 1, then Rhat is set to 1 and Neff
    set to (M-M0)*N, their limit values.

    Note: If N = 1 or M0 > M-2, Rhat = 0; Neff = 0.
    '''
    A = np.array(A)

    M, N = A.shape

    # Initialisation
    Rhat = 0
    Neff = 0

    # The convergence statistics can only be evaluated if there are multiple chains
    # and the chain length is greater than the burn in length

    if N > 1 and M > M0 + 1:
        Mt = M - M0

        # Chain mean and mean of means
        asub = A[M0:,:]
        ad = np.mean(asub,axis = 0)
        add = asub.mean()

        # Within group standard deviation
        ss = np.std(asub,axis = 0)

        # Between groups variance.
        dd = np.square(ad - add)
        B = (Mt*np.sum(dd))/(N-1)

        # Within groups variance.
        W = np.sum(np.square(ss))/N

        # V plus
        Vp = (Mt-1)*W/Mt + B/Mt

        # Convergence statistic, effective number of independent samples
        Rhat = np.sqrt(Vp/W)
        Neff = Mt*N*Vp/B

        Rhat = np.maximum(Rhat,1)
        Neff = np.minimum(Neff,Mt*N)

    return Rhat, Neff

"""### mcsums: Summary information from MC samples."""

def mcsums(A,M0,Q):
    '''
    mcsums: Summary information from MC samples.
    -------------------------------------------------------------------------
    KJ, LRW, PMH
    Version 2020-04-22

    -------------------------------------------------------------------------
    Inputs:
    A(M,N): An array that stores samples of size M x N.

    M0: Burn-in period with M > M0 >= 0.

    Q(nQ,1): Percentiles specifications, 0 <= Q(l) <= 100.

    Outputs:
    abar(n,1): Mean for each sample.

    s(n,1): Standard deviation for sample.

    aQ(nQ,n): Percentiles corresponding to Q.

    '''
    # Size of samples after burn-in
    A = np.array(A)

    M, N = A.shape

    m = (M - M0)*N

    # Initialise percentile vector
    nQ = Q.size
    aQ = np.zeros(nQ)

    # Samples from N parallel chains after burn-in period
    aaj = A[M0:, :]
    aaj = aaj.flatten()

    # Mean and standard deviation of samples
    abar = np.mean(aaj)
    s = np.std(aaj)

    # Percentiles of samples
    aQ = np.percentile(aaj,Q)

    return abar, s, aQ

"""### jumprwg: Jumping distribution for the Metropolis Hastings Gaussian random walk algorithm"""

def jumprwg(A, L):
    '''
    jumprwg: Jumping distribution for the Metropolis Hastings Gaussian random
    walk algorithm
    -------------------------------------------------------------------------
    KJ, LRW, PMH
    Version 2020-04-22
    -------------------------------------------------------------------------
    Inputs:
    A(n,N): Samples at the current iteration

    L(n,n): Cholesky factor of variance of parameter vector.

    Outputs:
    As(n,N): Proposed parameter array which is randomly sampled from the
    jumping distribution

    dp0: The difference between the logarithm of the jumping distribution
    associated with moving from A(:,j) to As(:,j) and that associated with
    moving from As(:,j) to A(:,j), up to an additive constant.
    log P0(a|as) - log P0(as|a)

    '''
    # Number of parameters and parallel chains
    n, N = A.shape

    # random draw from a Gaussian distribution
    e = np.random.normal(0, 1, size=(n,N))

    # proposed draw from a Gaussian distribution with mean A and variance LL'
    As = A + np.matmul(L,e)

    # For a Gaussian random walk, since log P0(a|as) = log P0(as|a), dp0 will always be zero
    dp0 = np.zeros(N)

    return As, dp0

"""### Cubic function and its first and second derivative"""

def fgh_cubic(alpha,t):
    '''
    -------------------------------------------------------------------------
    Cubic function and its first and second derivative
    -------------------------------------------------------------------------
    KJ, LRW, PMH
    Version 2020-04-22
    -------------------------------------------------------------------------
    Inputs:
    alpha(4,N):             Alpha parameters

    t(m,1):                 Times

    Outputs:
    f(m,N):                 Cubic function

    f1(m,N):                Derivative of cubic

    f2(m,N):                Second derivative of cubic
    '''

    # length of data and number of paramaters
    m = t.size

    # design matrix
    C = np.array([np.ones(m), t, t**2, t**3])

    # derivate info
    C1 = np.array([np.ones(m), 2*t, 3*t**2])
    C2 = np.array([2*np.ones(m), 6*t])

    # cubic and derivatives
    f = np.matmul(C.T,alpha)
    f1 = np.matmul(C1.T,alpha[1:])
    f2 = np.matmul(C2.T,alpha[2:])

    return f, f1, f2

"""### Log of the gaussian pdf"""

def ln_gauss_pdf_v(x,mu,sigma):
    '''
    -------------------------------------------------------------------------
    Log of the Gaussian pdf
    -------------------------------------------------------------------------
    KJ, LRW, PMH
    Version 2020-03-12
    --------------------------------------------------------------------------
    Inputs:
    x(m,1):                 Points at which pdf is to be evaluated

    mu:                     Mean of distribution

    sigma:                  Standard deviation of the distribution

    Output:
    logf:                   Log of the Gaussian pdf at x with mean mu and
                            std sigma
    '''


    try:
      # When inputs are high dimensional arrays/matrices
      xx = np.matlib.repmat(x,mu.shape[1],1)
      xx = xx.T
      logk = - np.log(2*math.pi)/2 - np.log(sigma)
      logp = -((xx - mu)**2)/(2*sigma**2)
      # Log of the Gaussian PDF
      logf = logk + logp

    except IndexError:
      # When inputs are vectors
      logk = - np.log(2*math.pi)/2 - np.log(sigma)
      logp = -((x - mu)**2)/(2*sigma**2)
      # Log of the Gaussian PDF
      logf = logk + logp

    return logf

"""### Target dist for noise and jitter posterior dist"""

def tar_at(at, y, x, m0w, s0w, m0t, s0t):
    '''
    -------------------------------------------------------------------------
    Target dist for noise and jitter posterior dist
    -------------------------------------------------------------------------
    KJ, LRW, PMH
    Version 2020-04-22
    --------------------------------------------------------------------------
    Inputs:
    at(n+2,N):              Parameters alpha, log(1/tau^2) and log(1/w^2)

    y(m,1):                 Signal

    x(m,1):                 time at which signal was recorded

    s0w and s0t:            prior estimates of tau and omega

    m0w and m0t:            degree of belief in prior estimates for tau and omega

    Output:
    T:                      Log of the posterior distribution
    '''

    # Size of parameter vector
    at = np.array(at)
    p = at.shape[0]

    # Number of alphas
    n = p - 2

    # Extract parameters
    alpha = at[0:n]
    phi1 = np.exp(at[-2])
    phi2 = np.exp(at[-1])
    taus = np.ones(phi1.shape)/np.sqrt(phi1)
    omegas = np.ones(phi2.shape)/np.sqrt(phi2)

    # Gamma priors for phis
    prior_phi1 = (m0t/2)*np.log(phi1) - phi1*m0t*s0t**2/2
    prior_phi2 = (m0w/2)*np.log(phi2) - phi2*m0w*s0w**2/2


    # function that evaluates the cubic function with user specified cubic parameters
    fun = lambda aa: fgh_cubic(aa,x)

    # cubic, expectation and variance
    [st,st1,st2] = fun(alpha)
    expect = st + 0.5*(taus**2)*st2
    vari = (taus**2)*(st1**2) + omegas**2

    # Likelihood
    lik = sum(ln_gauss_pdf_v(y,expect,np.sqrt(vari)))

    # Posterior
    T = lik + prior_phi1 + prior_phi2

    return T

"""###mcmcmh: Metrolopolis-Hasting MCMC algorithm generating N chains of length M for a parameter vector A of length n."""

def mcmcmh(M,N,M0,Q,A0, tar, jump):
    '''
    mcmcmh: Metrolopolis-Hasting MCMC algorithm generating N chains of length
    M for a parameter vector A of length n.

    For details about the algorithm please refer to:
    Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB.
    Bayesian data analysis. CRC press; 2013 Nov 1.
    -------------------------------------------------------------------------
    KJ, LRW, PMH
    Version 2020-04-22
    -------------------------------------------------------------------------
    Inputs:
    M: Length of the chains.

    N: Number of chains.

    M0: Burn in period.

    Q(nQ,1): Percentiles 0 <= Q(k) <= 100.

    A0(n,N): Array of feasible starting points: the target distribution
    evaluated at A0(:,j) is strictly positive.

    Outputs:

    S(2+nQ,n): Summary of A - mean, standard deviation and percentile limits,
    where the percentile limits are given by Q.

    aP(N,1): Acceptance percentages for AA calculated for each chain.

    Rh(n,1): Estimate of convergence. Theoretically Rh >= 1, and the closer
    to 1, the more evidence of convergence.

    Ne(n,1): Estimate of the number of effective number of independent draws.

    AA(M,N,n): Array storing the chains: A(i,j,k) is the kth element of the
    parameter vector stored as the ith member of the jth chain.
    AA(1,j,:) = A0(:,j).

    IAA(M,N): Acceptance indices. IAA(i,j) = 1 means that the proposal
    as(n,1) generated at the ith step of the jth chain was accepted so
    that AA(i,j,:) = as. IAA(i,j) = 0 means that the proposal as(n,1)
    generated at the ith step of the jth chain was rejected so that
    AA(i,j,:) = AA(i-1,j,:), i > 1. The first set of proposal coincide with
    A0 are all accepted, so IAA(1,j) = 1.
    '''
    A0 = np.array(A0)
    Q = np.array(Q)
    # number of parameters for which samples are to be drawn
    n = A0.shape[0]

    # number of percentiles to be evaluated
    nQ = Q.size


    # Initialising output arrays
    AA = np.empty((M, N, n))
    IAA = np.zeros((M, N))
    Rh = np.empty((n))
    Ne = np.empty((n))
    S = np.empty((2 + nQ, n))


    # starting values of the sample and associated log of target density
    aq = A0
    lq = tar(aq)

    # Starting values must be feasible for each chain
    Id = lq > -np.Inf

    if sum(Id) < N:
        print("Initial values must be feasible for all chains")
        return None


    # Run the chains in parallel
    for q in range(M):
        # draw from the jumping distribution and calculate
        # d = log P0(aq|as) - log P0(as|aq)

        asam, d = jump(aq)

        # log of the target density for the new draw as
        ls = tar(asam)

        # Metropolis-Hastings acceptance ratio
        rq = np.exp(ls - lq + d)

        # draws from the uniform distribution for acceptance rule
        uq = np.random.rand(N)
        # index of samples that have been accepted
        ind = uq < rq

        # updating the sample and evaluating acceptance indices
        aq[:, ind] = asam[:, ind]
        lq[ind] = ls[ind]
        IAA[q, ind] = 1

        # Store Metropolis Hastings sample
        AA[q, :, :] = np.transpose(aq)


    # acceptance probabilities for each chain
    aP = 100*np.sum(IAA,0)/M

    # Convergence and summary statistics for each of the n parameters
    for j in range(n):
        # test convergence
        RN = mcmcci(np.squeeze(AA[:,:,j]), M0)
        Rh[j] = RN[0]
        Ne[j] = RN[1]

        # provide summary information
        asq = np.squeeze(AA[:,:,j])
        SS = mcsums(asq,M0,Q)
        S[0,j] = SS[0]
        S[1,j] = SS[1]
        S[2:,j] = SS[2]

    return S, aP, Rh, Ne, AA, IAA

def main(datay,datax,m0w,s0w,m0t,s0t,Mc,M0,Nc,Q):

  at0 = np.array((1,1,1,1, np.log(1/s0w**2), np.log(1/s0t**2)))
  # function that evaluates the log of the target distribution at given parameter values
  tar = lambda at: tar_at(at, datay, datax, m0w, s0w, m0t, s0t)
  # function that evaluates the negative log of the target distribution to evaluate MAP estimates
  mapp = lambda at: -tar(at)

  res = minimize(mapp, at0)
  pars = res.x
  V = res.hess_inv
  L = np.linalg.cholesky(V)


  # Function that draws sample from a Gaussian random walk jumping distribution
  jump = lambda A: jumprwg(A, L)


  rr = np.random.normal(0,1,size=(6,Nc))

  A0 = numpy.matlib.repmat(pars.T,Nc,1).T + np.matmul(L,rr)


  sam = mcmcmh(Mc,Nc,M0,Q,A0,tar,jump)
  return 1/np.sqrt(np.exp(sam[0][0,-2:]))
