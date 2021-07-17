import numpy as np
import numpy as np
import scipy.linalg as linalg
from numba import njit

import statsmodels.api as sm
import pandas as pd 
from tabulate import tabulate

'''Part 1: compute covariances at all lags and log likelihood'''


def all_covariances(M, sigmas):
    """Use Fast Fourier Transform to compute covariance function between O vars up to T-1 lags.

    See equation (108) in appendix B.5 of paper for details.

    Parameters
    ----------
    M      : array (T*O*Z), stacked impulse responses of nO variables to nZ shocks (MA(T-1) representation) 
    sigmas : array (Z), standard deviations of shocks

    Returns
    ----------
    Sigma : array (T*O*O), covariance function between O variables for 0, ..., T-1 lags
    """
    T = M.shape[0]
    dft = np.fft.rfftn(M, s=(2 * T - 2,), axes=(0,))
    total = (dft.conjugate() * sigmas**2) @ dft.swapaxes(1, 2)
    return np.fft.irfftn(total, s=(2 * T - 2,), axes=(0,))[:T]


def log_likelihood(Y, Sigma, sigma_measurement=None):
    """Given second moments, compute log-likelihood of data Y.

    Parameters
    ----------
    Y       : array (Tobs*O)
                stacked data for O observables over Tobs periods
    Sigma   : array (T*O*O)
                covariance between observables in model for 0, ... , T lags (e.g. from all_covariances)
    sigma_measurement : [optional] array (O)
                            std of measurement error for each observable, assumed zero if not provided

    Returns
    ----------
    L : scalar, log-likelihood
    """
    Tobs, nO = Y.shape
    if sigma_measurement is None:
        sigma_measurement = np.zeros(nO)
    V = build_full_covariance_matrix(Sigma, sigma_measurement, Tobs)
    y = Y.ravel()
    return log_likelihood_formula(y, V)


'''Part 2: helper functions'''


def log_likelihood_formula(y, V):
    """Implements multivariate normal log-likelihood formula using Cholesky with data vector y and variance V.
       Calculates -log det(V)/2 - y'V^(-1)y/2
    """
    V_factored = linalg.cho_factor(V)
    quadratic_form = np.dot(y, linalg.cho_solve(V_factored, y))
    log_determinant = 2*np.sum(np.log(np.diag(V_factored[0])))
    return -(log_determinant + quadratic_form) / 2


@njit
def build_full_covariance_matrix(Sigma, sigma_measurement, Tobs):
    """Takes in T*O*O array Sigma with covariances at each lag t,
    assembles them into (Tobs*O)*(Tobs*O) matrix of covariances, including measurement errors.
    """
    T, O, O = Sigma.shape
    V = np.empty((Tobs, O, Tobs, O))
    for t1 in range(Tobs):
        for t2 in range(Tobs):
            if abs(t1-t2) >= T:
                V[t1, :, t2, :] = np.zeros((O, O))
            else:
                if t1 < t2:
                    V[t1, : , t2, :] = Sigma[t2-t1, :, :]
                elif t1 > t2:
                    V[t1, : , t2, :] = Sigma[t1-t2, :, :].T
                else:
                    # want exactly symmetric
                    V[t1, :, t2, :] = (np.diag(sigma_measurement**2) + (Sigma[0, :, :]+Sigma[0, :, :].T)/2)
    return V.reshape((Tobs*O, Tobs*O))


'''Part 3: Estimation'''

def build_M_sigma(par_est, par_est_names, Time, exo, data_targets, G):
    varN = 0 
    dX = None
    for k in par_est_names['rho']:
        dZ = par_est['rho'][k]**(np.arange(Time))
        jk = 0
        for j in data_targets:
            # if jk > 1:
            #     dX_temp = np.empty([Time,1])
            #     dX_temp[:,0] = G[j][exo[varN]] @ dZ
            #else:
            dX_temp = G[j][exo[varN]] @ dZ
            if jk == 0:
                 dX_specific_shock = dX_temp
            elif jk == 1:
                 dX_specific_shock = np.stack([ dX_specific_shock, dX_temp], axis=1)   
            else:
                dX_specific_shock = np.hstack([ dX_specific_shock, dX_temp[:,np.newaxis]])  
            jk += 1
        
        if dX is None:
            dX = dX_specific_shock
        elif varN == 1:
            dX = np.stack([dX, dX_specific_shock], axis=2)   
        else:
            dX = np.dstack([dX, dX_specific_shock])
             
        varN += 1

    sigmas = np.empty([len(par_est_names['sig'])])
    i = 0
     
    for k in par_est_names['sig']:
        #print(par_est['sig'][k] )
        sigmas[i] = par_est['sig'][k]
        i += 1 

    return dX, sigmas 

def constr_loglik(par_est, par_est_names, Y, Time, exo, data_targets, G, ss):
    """
    Returns multivariate normal log-likelihood where:
    par_est : parameters values for parameters to be estimated 
    par_est_names : list with parameter names 
    Y : data in numpy array 
    Time : Time horizon for jacobian 
    exo : variables assiocated with shocks      
    data_targets : list of strings with model names corresponding to data (i.e. GDP - logY)
    ss : ss dict 
    """
    
    # impulse response to persistent shock
    if par_est_names['internal']:
        ss.update({k : par_est[k] for k in par_est_names})
        G = jac.get_G(block_list, exo, unknowns, targets,  Time, ss, save=False, use_saved = True)

    dX, sigmas  = build_M_sigma(par_est, par_est_names, Time, exo, data_targets, G)
    Sigma = all_covariances(dX, sigmas)  

    # calculate log=likelihood from this
    return log_likelihood(Y, Sigma)


def FEVD(G, var, exo, hori, Time, rhos, sigmas):
    IRFs = {}
    for j in var:
        IRFs[j] = {}
        for k in exo:
            dExo = rhos[k]**(np.arange(Time)) * sigmas[k]
            IRFs[j][k] =  G[j][k] @ dExo
    
    
    
    FEVD = {}
    for j in var: 
        FEVD[j] = {}
        for i in exo:
            FEVD[j][i] = np.empty([hori]) * np.nan
        for h in range(hori):
            tot_var = 0 
            for i in exo:
                tot_var += np.sum(IRFs[j][i][:h+1]**2)
            for i in exo:        
                FEVD[j][i][h] = np.sum(IRFs[j][i][:h+1]**2 / tot_var)
                
    return FEVD

def Print_corrs(est_data, sim, lag):    
    moments_dict = {}
    moments_dict['std'] = {}
    moments_dict['Ycorr'] = {}
    moments_dict['autocorr'] = {}
    
    moments_dict['std']['data'] = {}
    moments_dict['Ycorr']['data'] = {}
    moments_dict['autocorr']['data'] = {}
    for k in est_data:
        moments_dict['std']['data'][k] =  np.std(est_data[k])
        corr = np.corrcoef(est_data[k], est_data['logY'])
        moments_dict['Ycorr']['data'][k] = corr[0,1]
        autocorr = sm.tsa.acf(est_data[k], nlags=lag, fft=True) 
        moments_dict['autocorr']['data'][k] = autocorr[lag]
     
    moments_dict['std']['Model (global)'] = {}    
    moments_dict['Ycorr']['Model (global)'] = {}
    moments_dict['autocorr']['Model (global)'] = {}
    for k in sim:
        moments_dict['std']['Model (global)'][k] =  np.std(sim[k])
        corr = np.corrcoef(sim[k], sim['logY'])
        moments_dict['Ycorr']['Model (global)'][k] =  corr[0,1]
        autocorr = sm.tsa.acf(sim[k], nlags=lag, fft=True) 
        moments_dict['autocorr']['Model (global)'][k] = autocorr[lag]
        
    
    df = pd.DataFrame(moments_dict['std'])
    print('Standard Deviations')
    print(tabulate(df.T, headers="keys"))
    
    print('Correlation with output')
    df = pd.DataFrame(moments_dict['Ycorr'])
    print(tabulate(df.T, headers="keys"))
    
    print('Autocorrelation')
    df = pd.DataFrame(moments_dict['autocorr'])
    print(tabulate(df.T, headers="keys"))
    
    return moments_dict