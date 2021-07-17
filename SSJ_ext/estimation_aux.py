"""
This file contains functions used to solve, estimate and display results.
"""

import numpy as np
from numba import njit, prange
import scipy.optimize as opt
import pandas as pd

from SSJ_ext import jacobian as jac
from SSJ_ext import estimation as ssj_est
 
"""Part 1. Simulation """


@njit
def arma_irf(ar_coeff, ma_coeff, T):
    """Generates shock IRF for any ARMA process """
    x = np.empty((T,))
    n_ar = ar_coeff.size
    n_ma = ma_coeff.size
    sign_ma = -1  # this means all MA coefficients are multiplied by -1 (this is what SW etc all have)
    for t in range(T):
        if t == 0:
            x[t] = 1
        else:
            ar_sum = 0
            for i in range(min(n_ar, t)):
                ar_sum += ar_coeff[i] * x[t - 1 - i]
            ma_term = 0
            if 0 < t <= n_ma:
                ma_term = ma_coeff[t - 1]
            x[t] = ar_sum + ma_term * sign_ma
    return x


@njit(parallel=True)
def simul_shock(dX, epsilon):
    dX_flipped = dX[::-1].copy()  # flip so we don't need to flip epsilon
    T = len(dX)
    T_simul = len(epsilon)
    Y = np.empty(T_simul - T + 1)
    for t in prange(T_simul - T + 1):
        Y[t] = np.vdot(dX_flipped, epsilon[t:t + T])

    return Y


def simulate(Gs, inputs, outputs, rhos, sigmas, T, T_simul):
    # Simulate outputs for AR(1) inputs with processes given by (rho, sigma)
    # Gs is the dict of G's 
    # inputs is the input list
    # outputs is the output list
    # T is the length of the impulse responses 
    # T_simul is simulation length
    epsilons = {i: np.random.randn(T_simul + T - 1) for i in inputs}
    simulations = {}

    for o in outputs:
        dXs = {i: sigmas[i] * (Gs[o][i] @ rhos[i] ** np.arange(T)) for i in inputs}
        simulations[o] = sum(simul_shock(dXs[i], epsilons[i]) for i in inputs)

    return simulations


"""Part 2. Estimation """


def jacobian(f, x0, f0=None, dx=1e-4):
    """Compute Jacobian of generic function."""
    if f0 is None:
        f0 = f(x0)
    n = x0.shape[0]
    Im = np.eye(n)
    J = np.empty(f0.shape + (n,))
    for i in range(n):
        J[..., i] = (f0 - f(x0 - dx * Im[i, :])) / dx
    return J


def hessian(f, x0, nfev=0, f_x0=None, dx=1e-4):
    """Compute Hessian of generic function."""
    n = x0.shape[0]
    Im = np.eye(n)

    # check if function value is given
    if f_x0 is None:
        f_x0 = f(x0)
        nfev += 1

    # compute Jacobian
    J = np.empty(n)
    for i in range(n):
        J[i] = (f_x0 - f(x0 - dx * Im[i, :])) / dx
        nfev += 1

    # compute the Hessian
    H = np.empty((n, n))
    for i in range(n):
        f_xi = f(x0 + dx * Im[i, :])
        nfev += 1
        H[i, i] = ((f_xi - f_x0) / dx - J[i]) / dx
        for j in range(i):
            jac_j_at_xi = (f(x0 + dx * Im[i, :] + dx * Im[j, :]) - f_xi) / dx
            nfev += 1
            H[i, j] = (jac_j_at_xi - J[j]) / dx - H[j, j]
            H[j, i] = H[i, j]

    return H, nfev


def get_normalized_data(ss, file, series):
    # load data. Note: data is *annualized* and *in percentages*
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("Q")

    crosswalk = {'y': 'Y', 'c': 'C', 'I': 'I', 'n': 'N', 'w': 'w', 'pi': 'pi', 'i': 'i'}

    # make quarterly
    for var in ['pi', 'i', 'y', 'c', 'I', 'n']:
        df[var] = df[var] / 4

    # convert quantities into percentage deviations from ss output
    df_out = df.copy()
    # for var in ['y', 'c', 'I', 'n', 'w']:
    #     if var in ss:
    #         df_out[var] *= ss[crosswalk[var]] / ss['Y']

    return df_out[series].values


def log_priors(x, priors_list):
    """This function computes a sum of log prior distributions that are stored in priors_list.
    Example: priors_list = {('Normal', 0, 1), ('Invgamma', 1, 2)}
    and x = np.array([1, 2])"""
    assert len(x) == len(priors_list)
    sum_log_priors = 0
    for n in range(len(x)):
        dist = priors_list[n][0]
        mu = priors_list[n][1]
        sig = priors_list[n][2]
        if dist == 'Normal':
            sum_log_priors += - 0.5 * ((x[n] - mu) / sig) ** 2
        elif dist == 'Uniform':
            lb = mu
            ub = sig
            sum_log_priors += - np.log(ub - lb)
        elif dist == 'Invgamma':
            alpha = (mu / sig) ** 2 + 2
            beta = mu * (alpha - 1)
            sum_log_priors += (-alpha - 1) * np.log(x[n]) - beta / x[n]
        elif dist == 'Invgamma_hs':
            s = mu
            v = sig
            sum_log_priors += (-v - 1) * np.log(x[n]) - v * s ** 2 / (2 * x[n] ** 2)
        elif dist == 'Gamma':
            theta = sig ** 2 / mu
            k = mu / theta
            sum_log_priors += (k - 1) * np.log(x[n]) - x[n] / theta
        elif dist == 'Beta':
            alpha = (mu * (1 - mu) - sig ** 2) / (sig ** 2 / mu)
            beta = alpha / mu - alpha
            sum_log_priors += (alpha - 1) * np.log(x[n]) + (beta - 1) * np.log(1 - x[n])
        else:
            raise ValueError('Distribution provided is not implemented in log_priors!')

    if np.isinf(sum_log_priors) or np.isnan(sum_log_priors):
        print(x)
        raise ValueError('Need tighter bounds to prevent prior value = 0')
    return sum_log_priors


def get_shocks_arma(x, shock_series):
    ix, ishock = 0, 0
    sigmas, arcoefs, macoefs = np.zeros((3, len(shock_series)))
    for shock_name, order in shock_series:
        sigmas[ishock] = x[ix]
        ix += 1
        if order >= 1:
            arcoefs[ishock] = x[ix]
            ix += 1
        if order >= 2:
            macoefs[ishock] = x[ix]
            ix += 1
        ishock += 1
    return sigmas, arcoefs, macoefs


def step1_est(G, arcoefs, macoefs, shock_series, outputs, T, T_irf, n_se, n_sh):
    """Compute the MA representation given G"""
    # Compute MA representation of outcomes in As
    As = np.empty((T_irf, n_se, n_sh))
    for i_sh in range(n_sh):

        arma_shock = arma_irf(np.array([arcoefs[i_sh]]), np.array([macoefs[i_sh]]), T)

        if np.abs(arma_shock[-1]) > 1e20:
            raise Warning('ARMA shock misspecified, leading to explosive shock path!')

        # store for each series
        shockname = shock_series[i_sh][0]
        for i_se in range(n_se):
            As[:, i_se, i_sh] = (G[outputs[i_se]][shockname] @ arma_shock)[:T_irf]

    return As


def step3_est(Sigma, y, sigma_o=None):
    To, O = y.shape
    loglik = (ssj_est.log_likelihood(y, Sigma, sigma_o) - (To * O * np.log(2 * np.pi)) / 2)
    return loglik


def loglik_f(x, data, outputs, shock_series, priors_list, T, G):
    T_irf = T - 20
    n_se, n_sh = len(outputs), len(shock_series)
    meas_error = np.zeros(n_se)  # set measurement error to zero

    # extract shock parameters from x; order: always sigma first, then AR coefs, then MA coefs
    sigmas, arcoefs, macoefs = get_shocks_arma(x, shock_series)

    # Step 1
    As = step1_est(G, arcoefs, macoefs, shock_series, outputs, T, T_irf, n_se, n_sh)

    # Step 2
    Sigma = ssj_est.all_covariances(As, sigmas)

    # Step 3
    llh = step3_est(Sigma, data, sigma_o=meas_error)

    # compute the posterior by adding the log prior
    log_posterior = llh + log_priors(x, priors_list)

    return log_posterior


def step1_estfull(x, shock_series, outputs, T, params, jac_info):
    """Compute the MA representation when we estimate model parameters"""

    # Update parameters
    n_params = len(params)
    x_params = x[-n_params:]

    # write new parameters into ss
    ss = jac_info['ss'].copy()
    ss.update({param: x_params[j] for j, param in enumerate(params)})

    # Compute model jacobian G
    G = jac.get_G(jac_info['block_list'], exogenous=jac_info['exogenous'], unknowns=jac_info['unknowns'],
                  targets=jac_info['targets'], T=jac_info['T'], ss=ss, outputs=outputs)

    T_irf = T - 20
    n_se, n_sh = len(outputs), len(shock_series)
    sigmas, arcoefs, macoefs = get_shocks_arma(x, shock_series)

    # Compute Ma representation
    return step1_est(G, arcoefs, macoefs, shock_series, outputs, T, T_irf, n_se, n_sh)


def estimate(outputs, data, x_guess, shock_series, priors_list, T, G=None, params=None, jac_info=None, sd=True,
             data_demean_f=False, **kwargs):
    if G is None and jac_info is None:
        raise ValueError('Need at least G or jac_info and params as input!')

    # If we do not need to compute the model jacobian G
    if G is not None:
        def objective(x):
            return - loglik_f(x, data, outputs, shock_series, priors_list, T, G)

    # If we do 
    else:
        # Store model jacobian when necessary
        n_params = len(params)
        last_x_params = [np.zeros(n_params)]
        last_G = [{}]

        def objective(x):
            # check whether we estimate the intercept of the data
            if data_demean_f is False:
                data_adj = data.copy()
            else:
                data_adj = data_demean_f(x, data)

            # Update parameters
            x_params = x[-n_params:]

            # Check whether params have changed or not since last iteration
            if not np.allclose(x_params, last_x_params[0], rtol=1e-12, atol=1e-12):

                # write new parameters into ss
                ss = jac_info['ss'].copy()
                ss.update({param: x_params[j] for j, param in enumerate(params)})

                # Compute model jacobian G
                G = jac.get_G(jac_info['block_list'], exogenous=jac_info['exogenous'], unknowns=jac_info['unknowns'],
                              targets=jac_info['targets'], T=jac_info['T'], ss=ss, outputs=outputs)

                # Store for later
                last_x_params[0] = x_params.copy()
                last_G[0] = G

            else:
                # if not, re-use the one from before
                G = last_G[0]

            # Compute log likelihood
            return - loglik_f(x, data_adj, outputs, shock_series, priors_list, T, G)
     
    # minimize objective
    #result = opt.minimize(objective, x_guess, **kwargs, method='SLSQP')
    result = opt.minimize(objective, x_guess, **kwargs)
    
    # Compute standard deviation if required
    if sd:
        H, nfev_total = hessian(objective, result.x, nfev=result.nfev, f_x0=result.fun)
        Hinv = np.linalg.inv(H)
        x_sd = np.sqrt(np.diagonal(Hinv))
    else:
        nfev_total = result.nfev
        x_sd = np.zeros_like(result.x)

    return result, x_sd, nfev_total


"""Part 3: Metropolis-Hastings """


def metropolis_hastings(lik_f, mode, Sigma, bounds, Nsim, Nburn, c):
    """ Metropolis Hastings algorithm: 
    lik_f: function that takes as input a vector of parameter and returns the log-likelihood    
    """
    # Initial parameters - draw from normal distribution centered at the mode
    check_bounds = False
    while check_bounds is False:
        x = np.random.multivariate_normal(mode, c * Sigma, 1)[0]
        check_bounds = checkbounds_mh_f(x, bounds)

    # Initialize vectors for simulation
    para_sim = np.zeros((mode.shape[0], Nsim))
    para_sim[:, 0] = x
    obj = lik_f(x)
    logposterior = obj * np.ones(Nsim)
    accept = 0

    # Iterate forward
    for i in range(1, Nsim):
    #if noisy:
        if i%100==0:
            print('Iteration:' , i)
        
        # Draw new parameters
        x = np.random.multivariate_normal(para_sim[:, i - 1], c * Sigma, 1)[0]
        check_bounds = checkbounds_mh_f(x, bounds)

        if check_bounds:

            # Compute log likelihood + log prior
            obj_new = lik_f(x)

            # Acceptance rate
            alpha = np.min((1, np.exp(obj_new - obj)))
            u = np.random.uniform()

            # Accept the new draw if u is less than acceptance threshold
            if u <= alpha:
                para_sim[:, i] = x
                obj = obj_new
                logposterior[i] = obj_new
                accept += 1
            else:
                para_sim[:, i] = para_sim[:, i - 1]
                logposterior[i] = obj

        else:
            para_sim[:, i] = para_sim[:, i - 1]
            logposterior[i] = obj

    # Drop initial periods
    para_sim = para_sim[:, Nburn:]
    logposterior = logposterior[Nburn:]

    # Compute acceptance rate, posterior mean and percentiles
    acceptancerate = accept / Nsim
    para_avg = np.mean(para_sim, 1)
    para_5 = np.percentile(para_sim, 5, 1)
    para_95 = np.percentile(para_sim, 95, 1)

    return para_sim, para_avg, para_5, para_95, logposterior, acceptancerate


def checkbounds_mh_f(x, bounds):
    check_bounds = True
    for i in range(len(x)):
        check_bounds = check_bounds & (x[i] > bounds[i][0]) & (x[i] < bounds[i][1])
    return check_bounds
