
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


@het(exogenous='Pi_p', policy=['l', 'k'], backward='k')
def Firm_prob1(k_p, Pi_p, k_grid, l_grid, z_grid, alpha, beta, p, delta, psi, r, w):
    """Single backward iteration step using endogenous gridpoint method for households with separable CRRA utility."""
    # this one is useful to do internally
    z = z_grid
    # uc_nextgrid = (beta * Pi_p) @ Va_p
    
    # labor
    # L = inv_labor_demand(w, beta, alpha, z, k_grid)
    
    #Invert dynamic foc 
    phi, dphi = get_phi_derive(k_p, k_grid[np.newaxis, :], psi)
    phi_prior, _ = get_phi_derive(k_grid[:, np.newaxis], k_grid[np.newaxis, :], psi)

    l = inv_labor_demand(w, beta, alpha, z[:, np.newaxis, np.newaxis], k_grid[np.newaxis, np.newaxis, :], p)

    E_costs = matrix_times_first_dim(Pi_p, ((1-delta) * pI + dphi * k_p / k_grid[np.newaxis, np.newaxis, :] - phi)/ (1+r))
    rhs = alpha * p * z[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :]**(alpha-1) * l**beta + E_costs 
    lhs = pI + phi_prior

    i, pi = lhs_equals_rhs_interpolate(rhs, lhs)
    k = utils.apply_coord(i, pi, k_grid)  
    l = inv_labor_demand(w, beta, alpha, z[:, np.newaxis, np.newaxis], k, p)

    
    y = z[:, np.newaxis, np.newaxis] * k**alpha * l**beta 
    phi = get_phi_derive(k, k_grid[np.newaxis, np.newaxis, :], psi)[0]
    inv = k - (1-delta) * k_grid[np.newaxis, np.newaxis, :]
    profit = p * y  - w*l - inv - phi
    
    return k, l, y, profit, inv

@njit 
def get_phi_derive(k_p, k, psi):
    phi  = psi /2 * (k_p/k-1)**2
    dphi = psi * (k_p/k-1)
    return phi, dphi
@njit
def inv_labor_demand(w, beta, alpha, z, k, p):
    return (w/(beta*p * z * k**alpha))**(1/(beta-1))

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


def HET_add_ons(): # add Pi_seed 
    z_grid, pi_e, Pi_p = utils.markov_rouwenhorst(rho=0.9, sigma=0.0001, N=2)
    return z_grid, pi_e, Pi_p

Firm_prob1_het = Firm_prob1.attach_hetinput(HET_add_ons)


alpha, beta, p, pI, delta, psi, r, w = 0.3, 0.4, 1, 1.2, 0.01, 2, 0.03, 0.6


k_grid = 5+ utils.agrid(amax=25, n=120)
l_grid = 0.1+ utils.agrid(amax=5, n=90)

z_grid, pi_e, Pi_p = utils.markov_rouwenhorst(rho=0.9, sigma=0.0001, N=2)
# k = z_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] + 0.05 * l_grid[np.newaxis, :, np.newaxis]

k = (( pI-(1-delta)/(1+r)) / (alpha*p*z_grid[:, np.newaxis, np.newaxis] * l_grid[np.newaxis, :, np.newaxis]**beta))**(1/(alpha-1)) + 0.001*k_grid[np.newaxis, np.newaxis, :]
k[k == np.inf] = k_grid[0]

ss = Firm_prob1_het.ss(k=k, Pi_p=Pi_p, k_grid=k_grid, l_grid=l_grid, 
                   z_grid=z_grid, alpha=alpha, beta=beta, p=p, delta=delta,
                   psi=psi, r=r, w=w, accelerated_it = False, noisy=True)


plt.plot(k_grid, ss['y'][0,10,:], ':')
plt.plot(k_grid, ss['y'][1,10,:], '--')
plt.show()

# impulse to demand shock 
Time = 300     
ttt = np.arange(0,Time)
dp = - 0.01  * 0.9**(np.arange(Time))
J      =   Firm_prob1_het.jac(ss, Time, ['p'])     

dY = J['Y']['p'] @ dp * 100 / ss['Y']
dK = J['K']['p'] @ dp * 100 / ss['K']
dL = J['L']['p'] @ dp * 100 / ss['L']

plt.plot(dY[:30], label ='Y')   
plt.plot(dK[:30], '--', label ='I')   
plt.plot(dL[:30], ':', label ='L')   
plt.plot(np.zeros(30), color = 'black', linestyle = '--')
plt.show()

print(ss['K'], ss['L'])

ss.update({'pI':pI, 'Z':1})


#%% Compare with simple jacobian IRF - no egm 

from solved_block import solved

@solved(unknowns=['K', 'L'], targets=['K_res', 'L_res'])
def firm(alpha, beta, p, pI, Z, psi, K, L):
    L_res = beta*p*Z*K**(alpha)*L**(beta-1) - w 
    phi  = psi /2 * (K(+1)/K-1)**2
    phi_prior  = psi /2 * (K/K(-1)-1)**2
    dphi = psi * (K(+1)/K-1)
    I = K - (1-delta) * K(-1)
    K_res = alpha*p*Z*K**(alpha-1)*L**beta + ((1-delta)*pI + dphi*K(+1)/K - phi)/(1+r) - (pI + phi_prior)
    Y = Z * K**alpha * L**beta
    profit = p * Y  - w*L - pI*I - phi
    return L_res, K_res, Y, profit, I 



def broyden_res(x):
    K,L = x   
    res1 = beta*p*Z*K**alpha * L**(beta-1) - w
    res2 = alpha*p*Z*K**(alpha-1) * L**beta + pI*(1-delta)/(1+r) - pI 
    return np.array([res1, res2])

Z = 1 
(Kss, Lss), _ = utils.broyden_solver(broyden_res, np.array([ss['K'],ss['L']]))

ss_simple = {'K' : Kss, 'L': Lss, 'alpha':alpha, 'beta':beta, 'Z':Z, 'p':p, 'pI':pI, 'psi':psi, 'pI':pI, 'Y': Z * Kss**alpha * Lss**beta}


J_simple      =   firm.jac(ss_simple, Time, ['p'])     

dY_simple = J_simple['Y']['p'] @ dp * 100 / ss_simple['Y']
dK_simple = J_simple['K']['p'] @ dp * 100 / ss_simple['K']
dL_simple   = J_simple['L']['p'] @ dp * 100 / ss_simple['L']

plt.plot(dY[:30])   
plt.plot(dY_simple[:30], ':')   
plt.plot(np.zeros(30), color = 'black', linestyle = '--')
plt.show()

plt.plot(dK[:30])   
plt.plot(dK_simple[:30], ':')   
plt.plot(np.zeros(30), color = 'black', linestyle = '--')
plt.show()


plt.plot(dL[:30])   
plt.plot(dL_simple[:30], ':')   
plt.plot(np.zeros(30), color = 'black', linestyle = '--')
plt.show()



