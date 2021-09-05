
import os
import sys
sys.path.append("..")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# %config InlineBackend.figure_format = 'retina'

# import SSJ_Org.utils as utils 
import SSJ_ext.utils as utils 
# from ..SSJ_ext import utils 

from SSJ_ext.het_block import het
import SSJ_ext.jacobian as jac
# from SSJ_ext import nonlinear
from SSJ_ext.simple_block import simple

from numba import vectorize, njit, guvectorize




@njit 
def get_phi_derive(x_p, x, par):
    val  = par /2 * (x_p/x-1)**2 * x
    dval1 = par * (x_p/x-1)
    dval2 = par / 2 * ((x_p/x)**2-1)
    return val, dval1, dval2


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


def HET_add_ons(rho, sigma, nZ): 
    z_grid, pi_e, Pi_p = utils.markov_rouwenhorst(rho, sigma, nZ)
    return z_grid, pi_e, Pi_p




# alpha, beta, p, pI, delta, psi, r, w = 0.3, 0.4, 1, 1.2, 0.01, 2, 0.03, 0.6
# m = 0.5 
# kappaV, ptheta, destr = 0.05 * w / m, 0.3, 0.1

# k_grid = 15   + utils.agrid(amax=10, n=120)
# l_grid = 1 + utils.agrid(amax=3, n=100)

# # z_grid, pi_e, Pi_p = utils.markov_rouwenhorst(rho=0.9, sigma=0.00001, N=2)
# rho, sigma, nZ = 0.9, 1e-05, 2

# cont_val_l = np.zeros([nZ, l_grid.size, k_grid.size])
# cont_val_l[:,:,:] = ((1-destr) * kappaV/m) / (1+r)
# cont_val_k = np.zeros([nZ, l_grid.size, k_grid.size])
# cont_val_k[:,:,:] = ((1-delta) * pI) / (1+r)

# ss = Firm_prob1_het.ss(cont_val_k=cont_val_k, cont_val_l=cont_val_l, k_grid=k_grid, l_grid=l_grid, 
#                     alpha=alpha, beta=beta, p=p, delta=delta, rho=rho, sigma=sigma, nZ=nZ,
#                     psi=psi, r=r, w=w, kappaV=kappaV, destr=destr, ptheta=ptheta, m=m, accelerated_it = False, noisy=True)


# plt.plot(k_grid, ss['y'][0,10,:], ':')
# plt.plot(k_grid, ss['y'][1,10,:], '--')
# plt.show()

# # impulse to demand shock 
# Time = 300     
# ttt = np.arange(0,Time)
# stime = 5
# dp = np.zeros(Time)
# dp[stime:] = - 0.01  * 0.9**(np.arange(Time-stime))
# J      =   Firm_prob1_het.jac(ss, Time, ['p'])     

# dY = J['Y']['p'] @ dp * 100 / ss['Y']
# dK = J['K']['p'] @ dp * 100 / ss['K']
# dL = J['L']['p'] @ dp * 100 / ss['L']

# plt.plot(dY[:30], label ='Y')   
# plt.plot(dK[:30], '--', label ='I')   
# plt.plot(dL[:30], ':', label ='L')   
# plt.plot(np.zeros(30), color = 'black', linestyle = '--')
# plt.show()

# print(ss['K'], ss['L'])

# ss.update({'pI':pI, 'Z':1})



# #%% Compare with simple jacobian IRF - no egm 

# from solved_block import solved

# @solved(unknowns=['K', 'L'], targets=['K_res', 'L_res'])
# def firm(alpha, beta, p, pI, Z, psi, K, L, ptheta, kappaV, destr, m, r):
#     theta  = ptheta /2 * (L(+1)/L-1)**2
#     theta_prior  = ptheta /2 * (L/L(-1)-1)**2
#     dtheta = ptheta * (L(+1)/L-1)    
    
#     L_res = beta*p*Z*K**(alpha)*L**(beta-1) + ((1-destr)*kappaV/m(+1) + dtheta*L(+1)/L - theta)/(1+r(+1)) - (w + kappaV/m + theta_prior)
    
#     phi  = psi /2 * (K(+1)/K-1)**2
#     phi_prior  = psi /2 * (K/K(-1)-1)**2
#     dphi = psi * (K(+1)/K-1)
#     I = K - (1-delta) * K(-1)
#     K_res = alpha*p*Z*K**(alpha-1)*L**beta + ((1-delta)*pI + dphi*K(+1)/K - phi)/(1+r(+1)) - (pI + phi_prior)
#     Y = Z * K**alpha * L**beta
#     profit = p * Y  - w*L - pI*I - phi
#     return L_res, K_res, Y, profit, I 



# def broyden_res(x):
#     K,L = x   
#     res1 = beta*p*Z*K**alpha * L**(beta-1) + (1-destr)*kappaV/m/(1+r) - (w+kappaV/m)
#     res2 = alpha*p*Z*K**(alpha-1) * L**beta + pI*(1-delta)/(1+r) - pI 
#     return np.array([res1, res2])


# Z = 1 
# (Kss, Lss), _ = utils.broyden_solver(broyden_res, np.array([ss['K'],ss['L']]))

# ss_simple = {'K' : Kss, 'L': Lss, 'alpha':alpha, 'beta':beta, 'Z':Z, 'p':p, 'pI':pI, 'psi':psi, 'pI':pI, 'Y': Z * Kss**alpha * Lss**beta, 
#              'ptheta':ptheta, 'kappaV':kappaV, 'm':m, 'r' : r, 'destr':destr}


# J_simple      =   firm.jac(ss_simple, Time, ['p'])     

# dY_simple = J_simple['Y']['p'] @ dp * 100 / ss_simple['Y']
# dK_simple = J_simple['K']['p'] @ dp * 100 / ss_simple['K']
# dL_simple   = J_simple['L']['p'] @ dp * 100 / ss_simple['L']

# plt.plot(dY[:30])   
# plt.plot(dY_simple[:30], ':')   
# plt.plot(np.zeros(30), color = 'black', linestyle = '--')
# plt.show()

# plt.plot(dK[:30])   
# plt.plot(dK_simple[:30], ':')   
# plt.plot(np.zeros(30), color = 'black', linestyle = '--')
# plt.show()


# plt.plot(dL[:30])   
# plt.plot(dL_simple[:30], ':')   
# plt.plot(np.zeros(30), color = 'black', linestyle = '--')
# plt.show()



