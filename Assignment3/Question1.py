### the following is copied from the code Prof. Sievers wrote and provided for this question: 
### the txt file is calling the date found and saved from https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/spectra/wmap_tt_spectrum_9yr_v5.txt


import numpy as np
import camb
from matplotlib import pyplot as plt
import time


def get_spectrum(pars,lmax=2000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt




pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

#plt.clf();
#plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*')
#plt.plot(wmap[:,0],wmap[:,1],'.')

cmb=get_spectrum(pars)
#plt.plot(cmb)
#plt.show()

###### end of Prof. Sievers' code ######

# defining chi-squared formula, equation taken from Lecture 4 notes 

def chi_squared(empiracal_data, fit_data, err):
    numerator=(empiracal_data-fit_data)**2
    denominator=err**2
    return numerator/denominator

# removing the first two points in the cmb data: 
cmb_data=cmb[2:len(wmap[:,0])+2]
wmap_chi_squared=np.sum(chi_squared(wmap[:,1], cmb_data, wmap[:,2]))
print(wmap_chi_squared) 
# this prints the value of 1588.2376465826746

# this is the expected value, around 1588 







