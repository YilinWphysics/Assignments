# once again start with Prof. Sievers' code: 

import numpy as np
import camb
from matplotlib import pyplot as plt
from wmap_camb_example import wmap, get_spectrum
import time


def get_spectrum_fixed_tau(pars,lmax=1500):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=0.05 
    As=pars[3] 
    ns=pars[4]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt


pars=np.asarray([65,0.02,0.1,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

#plt.clf();
#plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*')
#plt.plot(wmap[:,0],wmap[:,1],'.')
#cmb=get_spectrum(pars)
#plt.plot(cmb)
#plt.show()



##### end of Prof. Sievers' code #####

# optical depth fixed at 0.05 
# Newton's method / Levenberg-Marquardt minimizer 
# find best-fit values for the other parameters and their errors 

length=len(wmap[:,0]) # length of wmap data (any column will do)

def Newton_gradient(func, parameter):
    # defining gradient with respect to a speicific parameter for a given function; 
    gradient=np.zeros([length, len(parameter)]) # create zerio matrix
    for i in range(len(parameter)):
        parameter_update=parameter.copy()
        dx=parameter_update[i]*0.01
        parameter_update[i]=parameter_update[i]+dx
        func_left=func(parameter_update)[2:length+2]
        parameter_update[i]=parameter_update[i]-2*dx
        func_right=func(parameter_update)[2:length+2]
        gradient[:,i]=(func_left-func_right)/(2*dx)
    return gradient


convergence_criterion=1e-3 # common convergence criterion in ANSYS Fluent 
lambda_increase_factor=1000
lambda_decrease_factor=500

# defining Newton's method for when the residuals are already close to a minimum, i.e. already converging
# and when the residuals are large and are diverging, the Levenberg–Marquardt algorithm is employed for steep descend 
def Newton_Levenberg_Marquardt(empirical_data, fit_data, parameter, err, max_ite=100):
    lamb=0.001 #arbitrarily small 
    chi_squared=5000 #arbitrarily large
    gradient=Newton_gradient(fit_data, parameter)
    parameter_new=parameter.copy()
    for i in range(max_ite):
        guess=fit_data(parameter_new)[2:len(empirical_data)+2] # because in the data, same reason as Q1, the first two data points are to be discarded 
        residual=empirical_data-guess 
        chi_squared_new=np.sum((residual**2)/(err**2))
        # when chi_squared is already small enough and reducing: 
        if chi_squared_new < chi_squared and np.abs(chi_squared_new-chi_squared) < convergence_criterion: 
            parameter=parameter_new.copy()
            parameter = np.insert(parameter, 3, 0.05)
            print("The residuals are converging and meeting the convergence criterion of " + str(convergence_criterion) + r", with $\chi^{2}$ of " + str(chi_squared_new) + "\n")
            gradient=Newton_gradient(get_spectrum, parameter)
            noise=np.diag(1/(err**2))
            pcov=np.linalg.inv(np.dot(np.dot(gradient.transpose(), noise), gradient)) # definition of a covariance matrix 
            perr=np.sqrt(np.diag(pcov))
            break 

        # when chi_squared is diverging, increase lambda for steepest descend (Levenberg-Marquardt method)
        elif chi_squared_new > chi_squared: 
            print(r"$\chi^{2}$ diverging. Increase $\lambda$ by a factor of " + str(lambda_increase_factor) + "\n")
            lamb=lamb*lambda_increase_factor
            parameter_new=parameter.copy()

        else: 
            parameter=parameter_new.copy()
            lamb=lamb*lambda_decrease_factor
            gradient=Newton_gradient(fit_data, parameter)
            chi_squared=chi_squared_new
        residual=np.matrix(residual).transpose()
        gradient=np.matrix(gradient)
        Newton_lhs=gradient.transpose()@np.diag(1/(err**2))@gradient 
        lhs=Newton_lhs+np.diag(np.diag(Newton_lhs))*lamb
        rhs=gradient.transpose()@np.diag(1/(err**2))@residual
        dp=np.linalg.inv(lhs)@(rhs)
        for jj in range(parameter.size):
            parameter_new[jj]=parameter_new[jj]+dp[jj] # copied from class example Newton.py 
        print ("At iteration " + str({i}), r"the $\chi^{2}$ is " + str(chi_squared) + "\n")
    return parameter, pcov, perr 


# input initial values as given in Q1, in the following order: 
# Hubble constant, physixal baryon density, cold DM density, primordial amplitude of fluctuations, slope of primordial amplitude of fluctuations
H0=65 #km/s
wb_h2=0.02
wc_h2=0.1
As=2e-9
slope_ppl=0.96



pars=np.asarray([H0, wb_h2, wc_h2, As, slope_ppl])
pars,pcov,perr=Newton_Levenberg_Marquardt(wmap[:,1], get_spectrum_fixed_tau, pars, wmap[:,2])

print("The optimized parameters are the following: \n")
print(f"Hubble constant, H0, is {pars[0]} with an error of {perr[0]}")
print(f"Physical baryon density, wb_h2, is {pars[1]} with an error of {perr[1]}")
print(f"Cold DM density, wc_h2, is {pars[2]} with an error of {perr[2]}")
print(f"Optical depth, tau, is {pars[3]} with an error of {perr[3]}")
print(f"Primordial amplitude of fluctuations, As, is {pars[4]} with an error of {perr[4]}")
print(f"Slope of primordial amplitude of fluctuations, slope_ppl, is {pars[5]} with an error of {perr[5]}")

print(pcov)







