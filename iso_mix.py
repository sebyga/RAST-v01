
import numpy as np
def rast(isotherm_list,Lambda, C):
    if len(Lambda.shape) != 2:
        print('Lambda should be N x N array or matrix!')
        return
    elif Lambda.shape[0] != Lambda.shape[1]:
        print('Lambda should be N x N array or matrix!')
        return
    else:
        N = Lambda.shape[0]


def ln_gamma_i(x,Lamb, C, piA_RT):
    N = len(x)
    ln_gamma = []   # array N
    sum_xlam = []   # array N
    for ii in range(N):
        xlam_tmp = 0
        for jj in range(N):
            xlam_tmp = xlam_tmp + x[jj]*Lamb[ii,jj]
        sum_xlam.append(xlam_tmp)
        
        sum_xlam_ov_xlam = []
        for kk in range(N):
            sum_xlam_tmp = 0
            for ll in range(N):
                sum_xlam_tmp = sum_xlam_tmp + Lamb[kk,ll]*x[ll]
            sum_xlam_ov_xlam.append(Lamb[kk,ii])





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