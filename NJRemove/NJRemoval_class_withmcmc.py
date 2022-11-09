"""Script for analysing a measurand signal subject to noise and jitter effects
------------------------------------------------------------------------------
Author         LRW, KJ, PH   - National Physical Laboratory
               Bjoern Ludwig - Physikalisch-Technische Bundesanstalt
Project        EMPIR Met4FoF
------------------------------------------------------------------------------
Version        10/01/2020 - Creation
               25/02/2020 - Debugging
               05/03/2020 - Signal generation comparison with MATLAB code
               20/03/2020 - Debugging/combining with Bayesian work - see link
               31/03/2020 - Creating version of the code that is a class - for agentbased framework
               18/12/2020 - Agent version reviewed by Lizzie and updated accordingly
               08/11/2022 - Cleaned and blacked file
"""

import math

import numpy as np

import MCMCMH.mcmcm_decayexp


def DecayExpFunction(a, b, f, x):
    """decaying exponential function evaluated at x with parameters a, b and f"""
    return a * np.exp(-b * x) * np.sin(2 * np.pi * f * x)


def DecayExpFunction1der(a, b, f, x):
    """first derivative of the decaying exponential function

    This function provides the derivative of the decaying exponential function evaluated
    at x with parameters a, b and f.
    """
    return (
        a
        * np.exp(-b * x)
        * ((2 * np.pi * f) * np.cos(2 * np.pi * f * x) - b * np.sin(2 * np.pi * f * x))
    )


# Decaying exponential - second derivative
def DecayExpFunction2der(a, b, f, x):
    """second derivative of the decaying exponential function

    This function provides the second derivative of the decaying exponential function
    evaluated at x with parameters a, b and f.
    """
    return (
        a
        * np.exp(-b * x)
        * (
            (-np.power((2 * np.pi * f), 2)) * np.sin(2 * np.pi * f * x)
            - ((4 * b * np.pi * f) * np.cos(2 * np.pi * f * x))
            + b * np.sin(2 * np.pi * f * x)
        )
    )


class MCMCMH_NJ:
    """Bayesian Noise and jitter reduction algorithm

    MCMC used to determine the noise and jitter variances. Noise and jitter variances
    are then used in an iterative algorithm to remove the noise and jitter from the
    signal.
    """

    def __init__(self, fs, ydata, N, niter, tol, m0w, s0w, m0t, s0t, Mc, M0, Nc, Q):
        """Setting initial variables"""

        # variables for AnalyseSignalN and NJAlgorithm
        self.fs = fs
        self.ydata = ydata
        # self.xdata = xdata
        self.N = N
        self.niter = niter
        self.tol = tol
        self.m0w = m0w
        self.s0w = s0w
        self.m0t = m0t
        self.s0t = s0t
        self.Mc = Mc
        self.M0 = M0
        self.Nc = Nc
        self.Q = Q

    def AnalyseSignalN(self):
        """Analyse signal to remove noise and jitter

        Analyse signal to remove noise and jitter providing signal estimates with
        associated uncertainty. Uses normalised independent variable.
        """

        # Initialisation
        self.N = np.int64(self.N)  # Converting window length integer to int64 format
        m = np.size(self.ydata)  # Signal data length
        m = np.int64(m)  # Converting signal length to int64 format

        # Setting initial variables
        n = (self.N - 1) // 2
        # Covariance matrix for signal values
        Vmat = np.zeros((np.multiply(2, self.N) - 1, np.multiply(2, self.N) - 1))
        # Sensitivity vectors
        cvec = np.zeros(n, self.N)
        # Sensitivity matrix
        Cmat = np.zeros((self.N, np.multiply(2, self.N) - 1))
        # Initial signal estimate
        yhat0 = np.full((m, 1), np.nan)
        # Final signal estimate
        yhat = np.full((m, 1), np.nan)
        # Uncertainty information for estimates
        vyhat = np.full((m, self.N), np.nan)
        # Consistency matrix
        R = np.full((m, 1), np.nan)

        # Values of normalised independent variables
        datax = np.divide(np.arange(-n, n + 1), self.fs)

        outs = MCMCMH.mcmcm_decayexp.main(
            self.ydata,
            datax,
            self.m0w,
            self.s0w,
            self.m0t,
            self.s0t,
            self.Mc,
            self.M0,
            self.Nc,
            self.Q,
        )
        self.jitterSD = outs[0]
        self.noiseSD = outs[1]
        # Loop through indices L of window
        L = 0
        if np.size(self.ydata) > 10:
            # for L in range(1, m-self.N+1):
            # while
            # Index k of indices L of windows
            k = L + n
            # print(k)
            # Extract data in window
            datay = self.ydata[L : L + self.N]
            # Initial polynomial approximation
            p = np.polyfit(datax, datay, 3)
            pval = np.polyval(p, datax)
            yhat0[k] = pval[n]

            # Applying algorithm to remove noise and jitter
            [yhat[k], ck, vark, R[k]] = MCMCMH_NJ.NJAlgorithm(
                self, datax, datay, p, pval
            )
            print(yhat[k])
            # First n windows, start building the covariance matrix Vmat for the data
            if L < n + 1:
                Vmat[L - 1, L - 1] = vark

            # for windows n+1 to 2n, continue building the covariance matrix Vmat and
            # start storing the sensitivtity vectors ck in cvec
            elif L > n and L < np.multiply(2, n) + 1:
                Vmat[L - 1, L - 1] = vark
                cvec[L - n - 1, :] = ck

            # For windows between 2n+1 and 4n, continue to build Vmat and cvec, and
            # start building the sensitivity matrix Cmat from the sensitivity vecotrs.
            # Also, evaluate uncertainties for previous estimates.
            elif np.multiply(2, n) < L < np.multiply(4, n) + 2:

                Vmat[L - 1, L - 1] = vark
                # Count for building sensitivity matrix
                iC = L - np.multiply(2, n)
                # Start building sensitivity matrix from cvec
                Cmat[iC - 1, :] = np.concatenate(
                    (np.zeros((1, iC - 1)), cvec[0, :], np.zeros((1, self.N - iC))),
                    axis=None,
                )

                # Removing the first row of cvec and shift every row up one - creating
                # empty last row
                cvec = np.roll(cvec, -1, axis=0)
                cvec[-1, :] = 0
                # Update empty last row
                cvec[n - 1, :] = ck
                Cmatk = Cmat[0:iC, 1 : self.N - 1 + iC]

                # Continue building Vmat
                Vmatk = Vmat[1 : self.N - 1 + iC, 1 : self.N - 1 + iC]
                V = np.matmul(np.matmul(Cmatk, Vmatk), np.transpose(Cmatk))
                vhempty = np.empty((1, self.N - iC))
                vhempty[:] = np.nan
                # Begin building vyhat
                vyhat[L, :] = np.concatenate((vhempty, V[iC - 1, :]), axis=None)

            # For the remaining windows, update Vmat, Cmat and cvec. Continue to
            # evaluate the uncertainties for previous estimates.
            elif L > np.multiply(4, n) + 1:
                # Update Vmat
                Vmat = np.delete(Vmat, 0, axis=0)
                Vmat = np.delete(Vmat, 0, axis=1)
                Vmatzeros = np.zeros([Vmat.shape[0] + 1, Vmat.shape[1] + 1])
                Vmatzeros[: Vmat.shape[0], : Vmat.shape[1]] = Vmat
                Vmat = Vmatzeros
                Vmat[2 * self.N - 2, 2 * self.N - 2] = vark

                # Building updated Cmat matrix
                Cmat_old = np.concatenate(
                    (Cmat[1 : self.N, 1 : 2 * self.N - 1], np.zeros([self.N - 1, 1])),
                    axis=1,
                )
                Cmat_new = np.concatenate(
                    (np.zeros([1, self.N - 1]), cvec[0, :]), axis=None
                )
                Cmat = np.concatenate((Cmat_old, Cmat_new[:, None].T), axis=0)

                # Update cvec
                cvec = np.roll(cvec, -1, axis=0)
                cvec[-1, :] = 0

                # Continue building vyhat
                V = np.matmul(np.matmul(Cmat, Vmat), np.transpose(Cmat))
                vyhat[L, :] = V[self.N - 2, :]

        L += 1

        return yhat[k]

    def NJAlgorithm(self, datax, datay, p0, p0x):
        """Noise and Jitter Removal Algorithm

        Iterative scheme that preprocesses data to reduce the effects of
        noise and jitter, resulting in an estimate of the true signal along with its
        associated uncertainty.

        Refer paper for details: https://ieeexplore.ieee.org/document/9138266
        """

        # Initialisation
        iter_ = 0
        delta = np.multiply(2, self.tol)

        # Values of basis function at central point in window
        n = np.divide(self.N - 2, 2)
        k = np.int64(n + 1)
        t = np.array([np.power(datax[k], 3), np.power(datax[k], 2), datax[k], 1])
        # Design Matrix
        X = np.array(
            [
                np.power(datax, 3) + 3 * np.multiply(np.power(self.jitterSD, 2), datax),
                np.power(datax, 2) + np.power(self.jitterSD, 2),
                datax,
                np.ones(np.size(datax)),
            ]
        )
        X = X.T

        # Iterative algorithm
        while delta >= self.tol:
            # Increment number of iterations
            iter_ = iter_ + 1

            # Step 2 - Polynomial fitting over window
            pd = np.polyder(p0)
            pdx = np.polyval(pd, datax)

            # Step 4
            # Weight calculation
            w = np.divide(
                1,
                [
                    np.sqrt(
                        np.power(self.jitterSD, 2) * np.power(pdx, 2)
                        + np.power(self.noiseSD, 2)
                    )
                ],
            )
            w = w.T

            # Calculating polynomial coeffs
            Xt = np.matmul(np.diagflat(w), X)
            C = np.matmul(np.linalg.pinv(Xt), np.diagflat(w))
            datay = datay.T
            p1 = np.matmul(C, datay)
            p1x = np.polyval(p1, datax)

            # Step 5 - stabilise process
            delta = np.max(np.abs(p1x - p0x))
            p0 = p1
            p0x = p1x

            if iter_ == self.niter:
                print("Maximum number of iterations reached")
                break

        # Evaluate outputs
        c = np.matmul(t, C)
        yhat = np.matmul(c, datay)
        pd = np.polyder(p0)
        pdx = np.polyval(pd, datax[k])
        vk = np.power(self.jitterSD, 2) * np.power(pdx, 2) + np.power(self.noiseSD, 2)
        R = np.power(
            np.linalg.norm(np.matmul(np.diagflat(w), (datay - np.matmul(X, p0)))), 2
        )

        return yhat, c, vk, R


def random_gaussian_whrand(M, mu, sigma, istate1, istate2):
    """Generate random numbers from a Gaussian Distribution

    This function generates random numbers from a Gaussian Distribution using the
    Wichmann–Hill random number generator.

    Inputs:
    M:        Number of random numbers required
    mu:       Mean of the Gaussian
    sigma:    Std of the Gaussian
    istate1:  vector of 4 integers
    istate2:  vector of 4 different integers

    Output:
    xdist:    Random Gaussian vector of size M
    """
    mn = 0
    ndist = np.zeros(M)
    while mn < M:
        rr1 = whrand(istate1, 1)
        rr2 = whrand(istate2, 1)

        v1, istate1 = rr1
        v2, istate2 = rr2

        if mn < M:
            ndist[mn] = math.sqrt(-2 * math.log(v1)) * math.cos(2 * math.pi * v2)
            mn = mn + 1

        if mn < M:
            ndist[mn] = math.sqrt(-2 * math.log(v1)) * math.sin(2 * math.pi * v2)
            mn = mn + 1

        xdist = mu + sigma * ndist

    return xdist, istate1, istate2


def whrand(istate, N):
    """Generate uniform random numbers between 0 and 1

    This function generates uniform random numbers between 0 and 1 using the
    Wichmann–Hill random number generator.

    Inputs:
    istate:  vector of 4 integers
    N:       Number of random numbers required

    Outputs:
    r:       Vector uniform random numbers of size M
    istate:  Output vector of 4 integers
    """

    # Constants
    a = np.array([11600, 47003, 23000, 33000])
    b = np.array([185127, 45688, 93368, 65075])
    c = np.array([10379, 10479, 19423, 8123])
    d = np.array([456, 420, 300, 0]) + 2147483123

    r = np.zeros(N)
    for i in range(N):
        # Update states
        for j in range(4):
            istate[j] = a[j] * np.mod(istate[j], b[j]) - c[j] * np.fix(istate[j] / b[j])
            if istate[j] < 0:
                istate[j] = istate[j] + d[j]

        # Evaluate random number
        w = np.sum(np.divide(istate, d))
        r[i] = np.remainder(w, 1)

    return r, istate
