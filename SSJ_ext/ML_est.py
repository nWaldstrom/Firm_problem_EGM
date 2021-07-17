from SSJ_ext import estimation as est
import numpy as np 
 

'''Log-likelihood estimation'''

# penalty function
def penalty(par_est, par_est_names):   
    pen = 0 
    
    for k in par_est_names['rho']:
        x = par_est['rho'][k]
        pen_ = 0
        if x < 0:
            pen_ = abs(x) * 10 
            x = 0.00001
        if x > 1:
            pen_ = (abs(x)-1) * 10 
            x = 0.990  
        pen += pen_   
        par_est['rho'][k] = x

    for k in par_est_names['sig']:
        x = par_est['sig'][k]
        pen_ = 0
        if x < 0:
            pen_ = abs(x) * 10 
            x = 0.00001        
        pen += pen_
        par_est['sig'][k] = x
    
    return par_est, pen

def internal_par_bounds(x, bounds, par_est_names):
    pen = 0 
    eps = 1e-04
    if x < bounds[0]:
        pen = (abs(x)-bounds[0]) * 10 
        x = bounds[0] + eps
    if x > bounds[1]:
        pen = (abs(x)-bounds[1]) * 10 
        x = bounds[1] - eps        
    return x, pen

# read optimizer' guesses x into three dicts for rho, sigma and internal parameters.
# def x2dict(x, N_series, par_est_names):
#     # initalize stuff 
#     par_est = {}
#     par_est['rho'] = {} 
#     par_est['sig'] = {} 
#     par_est['internal'] = {} 
#     i = 0 
    
#     for k in par_est_names['rho']:
#         par_est['rho'][k] = x[i]
#         i += 1 
#     i = 0 
#     for k in par_est_names['sig']:
#         par_est['sig'][k] = x[N_series+i]
#         i += 1 
        
#     if par_est_names['internal']:
#         i = 0 
#         for k in par_est_names['internal']:
#             par_est['internal'][k] = x[2*N_series+i]
#             i += 1     
#     return par_est

def x2dict(x, N_series, par_est_names):
    # initalize stuff 
    par_est = {}
    par_est['rho'] = {} 
    par_est['sig'] = {} 
    par_est['internal'] = {} 
    i = 0 
    
    for k in par_est_names['rho']:
        par_est['rho'][k] = x[i]
        i += 1 
    i = 0 
    for k in par_est_names['sig']:
        par_est['sig'][k] = x[N_series+i]
        i += 1 
        
    if par_est_names['internal']:
        i = 0 
        for k in par_est_names['internal']:
            par_est['internal'][k] = x[2*N_series+i]
            i += 1     
    return par_est


# minimizer       
def max_like_obj(x, N_series, par_est_names, Y, G, ss_est, bounds_int_pars, exo, data_targets, Time):       

    # unpack from x 
    par_est = x2dict(x, N_series, par_est_names)   

    par_est, pen = penalty(par_est, par_est_names)

    # par_type = 'rho'
    # for k in par_est_names['rho']:
    #     par_est['rho'][k], pen_ = penalty(par_est['rho'][k], par_type)
    #     pen += pen_    
    # par_type = 'sig'
    # for k in par_est_names['sig']:
    #     par_est['sig'][k], pen_ = penalty(par_est['sig'][k], par_type)
    #     pen += pen_
    
    for k in par_est_names['internal']:
        par_est['internal'][k], pen_ = internal_par_bounds(par_est['internal'][k], bounds_int_pars[k], par_est_names)
        pen += pen_    
    
    #print(par_est['rho'])
    #args_dict = args[0]
    #Y = args_dict['Y']
    #G = args_dict['G']

    
    #print(rho_mup, rho_r, rho_beta, sigma_mup, sigma_r, sigma_beta)
    try:
        loglik = est.constr_loglik(par_est, par_est_names, Y, Time, exo, data_targets, G, ss_est)
    except:
        loglik = - 1e+09
    
    obj = -loglik + abs(loglik) * pen # we minimize         
    
    return obj  