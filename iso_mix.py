#%% Importing
# numericals
import numpy as np
import pandas as pd

# scipy
from scipy.integrate import trapz
from scipy.optimize import minimize
from scipy.optimize import shgo
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing

# files handling
import os
import pickle

# Graph
import matplotlib.pyplot as plt

#%% Single Isotherm 1: isotherm models
def Arrh(T, dH, T_ref):
    exp_term = np.exp(np.abs(dH)/8.3145*(1/T-1/T_ref))
    return exp_term
    
def Lang(par,P):
    bP = par[1]*P
    deno = 1+ bP
    nume = par[0]*bP
    q = nume/deno
    return q

def Quad(par,P):
    bP = par[1]*P
    dPP = par[2]*P
    deno = 1+ bP + dPP
    nume = par[0]*(2*bP + dPP)
    q = nume/deno
    return q

def DSLa(par,P):
    nume1 = par[0]*par[1]*P
    deno1 = 1+par[1]*P
    nume2 = par[2]*par[3]*P
    deno2 = 1+par[3]*P
    q = nume1/deno1 + nume2/deno2
    return q    

iso_fn_candi = [Lang,Quad, DSLa]
iso_fn_candi_str = ['Lang','Quad', 'DSLa']
iso_par_num = [2,3,4]

#%% Single Isothemr 2: Objective function 
def iso2err(par,P,q,iso_fn):
    par_arr = np.array(par)
    is_nega = par_arr<0
    penaltyy = np.sum(par_arr[is_nega]**2)*50
    par_arr[is_nega] = 0
    
    diff = (iso_fn(par_arr,np.array(P)) - np.array(q))/(np.array(q)+1E-3)
    err_sum = np.sum(diff**2)
    return err_sum + penaltyy

#%% Single Isothemr 3: Fitting function with a single model
method_list = ['Nelder-mead','Powell','COBYLA','differential_evolution']
def find_par(isofn, n_par, P,q, methods):
    p_arr = np.array(P)
    q_arr = np.array(q)
    obj_fn = lambda par: iso2err(par,p_arr,q_arr,isofn)
    optres_fun = []
    optres_x = []
    for me in methods: 
        if me =='shgo':
            bounds = np.zeros([n_par,2])
            bounds[:,1] = 5
            optres_tmp = shgo(obj_fn,bounds,)
        elif me == 'differential_evolution':
            bounds = np.zeros([n_par,2])
            bounds[:,1] = 5
            optres_tmp = differential_evolution(obj_fn, bounds,)
        elif me == 'dual_annealing':
            bounds = np.zeros([n_par,2])
            bounds[:,1] = 5
            optres_tmp = dual_annealing(obj_fn, bounds,)
        else:
            x0 = 2*np.ones(n_par)  # INITIAL GUESS !!!
            x0[0] = q[-1]
            optres_tmp = minimize(obj_fn,x0,method = me)
        optres_fun.append(optres_tmp.fun)
        optres_x.append(optres_tmp.x)
    bestm = np.argmin(optres_fun)
    par_sol = optres_x[bestm]
    fn_sol = optres_fun[bestm]
    return par_sol, fn_sol, optres_x, optres_fun

#%% Single Isotherm 4: Fitting with diff. isotherm models
def best_isomodel(P,q,tol = 4E-4):
    optfn_list = []
    optx_list = []
    for n_par,iso_fn in zip(iso_par_num, iso_fn_candi):
        parsol_tmp, fnsol_tmp, _,_ = find_par(iso_fn, n_par,P,q,method_list)
        optx_list.append(parsol_tmp)
        optfn_list.append(fnsol_tmp)
        if fnsol_tmp/len(q) <= tol:
            break
    #print(optfn_list)
    argMIN = np.argmin(np.array(optfn_list))
    x_best = np.array(optx_list)[argMIN]
    iso_best = lambda pp: iso_fn_candi[argMIN](x_best, pp)
    str_best = iso_fn_candi_str[argMIN]
    fnval_best = np.array(optfn_list)[argMIN]
    return iso_best, x_best, str_best, fnval_best

#%% RAST 1: (ln \gamma)
def ln_gamma_i(x,Lamb, C, piA_RT):
    N = len(x)
    ln_gamma = np.zeros(N)   # array N
    sum_xlam = []   # array N
    sum_xlam_ov_xlam = []
    exp_term = []
    for ii in range(N):
        xlam_tmp = 0
        for jj in range(N):
            xlam_tmp = xlam_tmp + x[jj]*Lamb[ii,jj]
        sum_xlam.append(xlam_tmp)
        sum_xlam_xlam = 0
        for kk in range(N):
            sum_xlam_tmp = 0
            for ll in range(N):
                sum_xlam_tmp = sum_xlam_tmp + Lamb[kk,ll]*x[ll]
            sum_xlam_xlam = sum_xlam_xlam + Lamb[kk,ii]*x[kk]/sum_xlam_tmp
        sum_xlam_ov_xlam.append(sum_xlam_xlam)
        exp_term_tmp = 1-np.exp(-C*piA_RT)
        exp_term.append(exp_term_tmp)
        all_term_tmp = (1-np.log(xlam_tmp)-sum_xlam_xlam)*exp_term_tmp
        ln_gamma[ii] = all_term_tmp
    return ln_gamma

#%% RAST 2: RAST mixture isotherm model
def rast(isotherm_list,P_i,T, Lamb, C):
    if len(Lamb.shape) != 2:
        print('Lambda should be N x N array or matrix!')
        return
    elif Lamb.shape[0] != Lamb.shape[1]:
        print('Lambda should be N x N array or matrix!')
        return
    else:
        N = Lamb.shape[0]
    
    def spreading_pressure(iso, P_max):
        P_ran = np.linspace(0.0001,P_max)
        q_ov_P = iso(P_ran)/P_ran
        spr_P = trapz(q_ov_P, P_ran)
        return spr_P
    iso_list = []
    iso_spr = []
    for isoo in isotherm_list:
        iso_tmp = lambda pp: isoo(pp, T)
        iso_spr_tmp = lambda ppp: spreading_pressure(iso_tmp, ppp)
        iso_list.append(iso_tmp)
        iso_spr.append(iso_spr_tmp)

    def spreading_P_err(x_N_piART):
        xx_first = x_N_piART[:N-1]
        xx_last = [1-np.sum(xx_first)]
        xx = np.concatenate([xx_first, xx_last])
        rms_err = 0
        for ii in range(N):
            if xx[ii] <0.0001:
                rms_err = rms_err+50*xx[ii]**2
                xx[ii] = 0.0001
            elif xx[ii] > 0.999:
                rms_err = rms_err+50*(xx[ii]-1)**2
                xx[ii] = 0.9999
        
        spr_P = x_N_piART[-1]
        ln_gam = ln_gamma_i(xx,Lamb, C, spr_P,)
        gamm = np.exp(ln_gam)
        Po_i = P_i/gamm/xx
        spr_P_new = np.zeros(N)
        for ii in range(N):
            spr_P_tmp = iso_spr[ii](Po_i[ii])
            spr_P_new[ii] = spr_P_tmp
        rms_err = np.sum((spr_P_new - spr_P)**2)
        return rms_err
    
    y_i = P_i/np.sum(P_i)
    x_init = P_i/np.sum(P_i)
    x_init = x_init[:-1]
    
    piA_RT_list = []
    qm = []
    #theta = []
    bP = []
    for iso,pp in zip(isotherm_list,P_i):
        P_ran = np.linspace(0.0001,pp)
        q_P = iso(P_ran, T)/P_ran
        piA_RT_tmp = trapz(q_P,P_ran)
        piA_RT_list.append(piA_RT_tmp)
        q_max = iso(1E8, T)
        theta_tmp = q_P[-1]*pp/q_max
        bP_tmp = theta_tmp/(1-theta_tmp)
        bP.append(bP_tmp)
        #theta.append(theta_tmp)
        qm.append(q_max)
    bP_sum = np.sum(bP)
    q_extended = np.array(qm)*np.array(bP)/(1+ bP_sum)
    x_extended = q_extended/np.sum(q_extended)
    x_ext_init = x_extended[:-1]

    opt_list = []
    opt_x_list = []
    opt_fn_list = []
    for spr_P0 in piA_RT_list:
        x0 = np.concatenate([x_init, [spr_P0]])
        optres_tmp = minimize(spreading_P_err, x0, method = 'Nelder-mead')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-6:
            #print(optres_tmp.fun)
            #print('YEAH')
            break
        optres_tmp = minimize(spreading_P_err, x0, method = 'Powell')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
        optres_tmp = minimize(spreading_P_err, x0, method = 'COBYLA')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
    for spr_P0 in piA_RT_list:
        x0 = np.concatenate([x_ext_init, [spr_P0]])
        optres_tmp = minimize(spreading_P_err, x0, method = 'Nelder-mead')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
        optres_tmp = minimize(spreading_P_err, x0, method = 'Powell')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
        optres_tmp = minimize(spreading_P_err, x0, method = 'COBYLA')
        opt_list.append(optres_tmp)
        opt_x_list.append(optres_tmp.x)
        opt_fn_list.append(optres_tmp.fun)
        if optres_tmp.fun < 1E-2:
            break
    #print(opt_fn_list)
    arg_min = np.argmin(opt_fn_list)
    x_re = np.zeros(N)
    x_re[:-1] = opt_list[arg_min].x[:-1]
    x_re[-1] = np.min([1- np.sum(x_re[:-1],0)])
    piA_RT_re = opt_list[arg_min].x[-1]
    ln_gam_re = ln_gamma_i(x_re,Lamb, C, piA_RT_re)
    gamma_re = np.exp(ln_gam_re)
    #print(iso_spr[0](P_i[0]/optres_tmp.x[0]/gamma_re[0]))
    #print(iso_spr[1](P_i[1]/(1- optres_tmp.x[0])/gamma_re[1]))
    arg_0 = x_re == 0
    arg_non0 = arg_0 == False
    P_pure = np.zeros(N)
    P_pure[arg_non0] = np.array(P_i)[arg_non0]/x_re[arg_non0]/gamma_re[arg_non0]
    #P_pure[arg_0] = np.array(P_i)[arg_0]/x_re[arg_0]/gamma_re[arg_0]
    #P_pure[arg_0] = 0
    q_pure = np.zeros(N)
    for ii in range(N):
        q_pure[ii] = iso_list[ii](P_pure[ii])
    q_tot = 1/(np.sum(x_re/q_pure))
    q_return = q_tot*x_re
    return q_return

#%% Main Test 1: RAST test
if __name__ == '__main__':
    R_gas = 8.3145
    Arrh =  lambda dH,T,Tref: np.exp(abs(dH/R_gas)*(1/T - 1/Tref))
    def Lang_PTdH(P,T,par,dH,Tref):
        bP = par[1]*P*Arrh(dH,T,Tref)
        numer = par[0]*bP
        denom = 1 + bP
        return numer/denom
    lang1 = lambda P,T: Lang_PTdH(P,T,[3, 0.1],10, 300)
    lang2 = lambda P,T: Lang_PTdH(P,T,[1, 0.5],20, 300)
    P_partial = np.array([2,0.1])
    Lambda_test = np.array([[1,0.8],[0.8,1]])
    C_test = 1
    rast_test_res = rast([lang1,lang2],P_partial,300,Lambda_test,C_test)
    print(rast_test_res)

    print(rast_test_res[0])
    print(rast_test_res[1])

    #%% Main Test 2: Single isotherm fitting error
    # Test the iso2err
    p_test = np.linspace(0,50, 51)
    q_test = 3*0.1*p_test/(1+0.1*p_test)
    err_test1 = iso2err([3,0.1], p_test,q_test, Lang)
    err_test2 = iso2err([3,0.1,0.1], p_test,q_test, Quad)
    err_test3 = iso2err([3,0.1,0.1,0.001], p_test,q_test, DSLa)
    print('[FITTING ERRORS]')
    print('[for Lagnmuir dummy data]')
    print()
    print('Langmuir:         ', err_test1)
    print('Quadratic:        ',err_test2)
    print('Dualsite Langmuir:', err_test3)

    #%% Main Test 3: Single isotherm model fitting
    #iso_fn_candi = [Lang,Quad, DSLa]
    #iso_fn_candi_str = ['Lang','Quad', 'DSLa']
    #iso_par_num = [2,3,4]
    parsol_test, fnsol_test,_,_ = find_par(iso_fn_candi[0],iso_par_num[0],p_test,q_test, method_list)

    isob_test, xb_test, strb_test, fvalb_test =best_isomodel(p_test,q_test)
    print('Function:', )
    print(isob_test)
    print('Parameters:',xb_test)
    print('Model (string): ',strb_test)


