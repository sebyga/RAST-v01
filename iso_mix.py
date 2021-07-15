
import numpy as np
from scipy.integrate import trapz
def rast(isotherm_list,P_i,T, Lamb, C):
    if len(Lamb.shape) != 2:
        print('Lambda should be N x N array or matrix!')
        return
    elif Lamb.shape[0] != Lamb.shape[1]:
        print('Lambda should be N x N array or matrix!')
        return
    else:
        N = Lamb.shape[0]
    def spreading_P_err(x_N_piART):

    y_i = P_i/np.sum(P_i)
    x_init = P_i/np.sum(P_i)
    piA_RT_list = []
    for xx,iso,pp in zip(x_init,isotherm_list,P_i):
        P_ran = np.linspace(0,pp)
        q_P = iso(P_ran, T)/P_ran
        piA_RT_tmp = trapz(q_P,P_ran)
        piA_RT_list.append(piA_RT_tmp)
        
    ln_gam = ln_gamma_i(x_init, Lamb,C, piA_RT_list[0])


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
        exp_term_tmp = 1-np.exp(C*piA_RT)
        exp_term.append(exp_term_tmp)
        all_term_tmp = (1-np.log(xlam_tmp)-sum_xlam_xlam)*exp_term_tmp
        ln_gamma[ii] = all_term_tmp
    return ln_gamma

        
            





if __name__ == '__main__':
    import numpy as np
    R_gas = 8.3145
    Arrh =  lambda dH,T,Tref: np.exp(abs(dH/R_gas)*(1/T - 1/Tref))
    def Lang(P,T,par,dH,Tref):
        bP = par[1]*P*Arrh(dH,T,Tref)
        numer = par[0]*bP
        denom = 1 + bP
        return numer/denom
    lang1 = lambda P,T: Lang(P,T,[3, 0.1],10, 300)
    lang2 = lambda P,T: Lang(P,T,[2, 0.2],10, 300)