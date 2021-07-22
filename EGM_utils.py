
import numpy as np
import SSJ_ext.utils as utils 
from numba import vectorize, njit, guvectorize


@njit 
def get_gamma_derive(x_p, x, gamma):
    fgamma  = gamma /2 * (x_p/x-1)**2 
    dfgamma1 = gamma * (x_p/x-1) 
    dfgamma2 = gamma /2 * (x_p/x-1)**2 - gamma * (x_p/x-1) * x_p/x
    return fgamma, dfgamma1, dfgamma2

@njit
def inv_R_demand(pR, theta, alpha, z, n, p):
    return (pR/(theta*p*z * n**alpha))**(1/(theta-1))

def matrix_times_first_dim(A, X):
    """Take matrix A times vector X[:, i1, i2, i3, ... , in] separately
    for each i1, i2, i3, ..., in. Same output as A @ X if X is 1D or 2D"""
    # flatten all dimensions of X except first, then multiply, then restore shape
    return (A @ X.reshape(X.shape[0], -1)).reshape(X.shape)

@guvectorize(['void(float64[:], float64[:,:], uint32[:], float64[:])'], '(ni),(ni,nj)->(nj),(nj)')
def lhs_equals_rhs_interpolate(lhs, rhs, iout, piout):
    """
    Given lhs (i) and rhs (i,j), for each j, find the i such that

    lhs[i] > rhs[i,j] and lhs[i+1] < rhs[i+1,j]
    
    i.e. where given j, lhs == rhs in between i and i+1.
    
    Also return the pi such that 

    pi*(lhs[i] - rhs[i,j]) + (1-pi)*(lhs[i+1] - rhs[i+1,j]) == 0

    i.e. such that the point at pi*i + (1-pi)*(i+1) satisfies lhs == rhs by linear interpolation.

    If lhs[0] < rhs[0,j] already, just return u=0 and pi=1.

    ***IMPORTANT: Assumes that solution i is monotonically increasing in j
    and that lhs - rhs is monotonically decreasing in i.***
    """
    ni, nj = rhs.shape
    assert len(lhs) == ni

    i = 0
    for j in range(nj):
        while True:
            if lhs[i] < rhs[i, j]:
                break
            elif i < nj - 1:
                i += 1
            else:
                break

        if i == 0:
            iout[j] = 0
            piout[j] = 1
        else:
            iout[j] = i - 1
            err_upper = rhs[i, j] - lhs[i]
            err_lower = rhs[i - 1, j] - lhs[i - 1]
            piout[j] = err_upper / (err_upper - err_lower)


def HET_add_ons(rho, sigma, NZ): 
    z_grid, pi_e, Pi_p = utils.markov_rouwenhorst(rho, sigma, NZ)
    return z_grid, pi_e, Pi_p


