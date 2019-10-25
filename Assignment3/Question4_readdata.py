import numpy as np 
import matplotlib.pyplot as plt 

parameters=np.loadtxt("Question4_datafile.txt", delimiter=",")
H0=parameters[:,0]
wb_h2=parameters[:,1]
wc_h2=parameters[:,2]
tau=parameters[:,3]
As=parameters[:,4]
slope_ppl=parameters[:,5]
chisq=parameters[:,6]


def gaussian(x, mu, sigma, N):
    return 1/(sigma*np.sqrt(2*np.pi*N))*np.exp(-0.5*(((x-mu)/sigma)**2/N))


tau_value=0.0544
tau_sigma=0.0073


figure, (ax1, ax2, ax3, ax4, ax5, ax6, ax7)=plt.subplots(7, 1, sharex=True) 
ax1.plot(H0)
ax1.set_ylabel("H0")
ax2.plot(wb_h2)
ax2.set_ylabel("wb_h2")
ax3.plot(wc_h2)
ax3.set_ylabel("wc_h2")
ax4.plot(tau)
ax4.set_ylabel("tau")
ax5.plot(As)
ax5.set_ylabel("As")
ax6.plot(slope_ppl)
ax6.set_ylabel("slope_ppl")
ax7.plot(chisq)
ax7.set_ylabel("chisq")
ax7.set_xlabel("Steps")
#plt.show()
figure.savefig("Question4_MCMC_chain.png")


burn_in_region_steps=1000 # read off from the plots, should be ~ 190; but remove a bit more just in case 
params_selected_region=parameters[burn_in_region_steps:,]
pars=np.mean(params_selected_region,axis=0)
perr=np.std(params_selected_region,axis=0)


print("The optimized parameters are the following: \n")
print(f"Hubble constant, H0, is {pars[0]} with an error of {perr[0]}")
print(f"Physical baryon density, wb_h2, is {pars[1]} with an error of {perr[1]}")
print(f"Cold DM density, wc_h2, is {pars[2]} with an error of {perr[2]}")
print(f"Optical depth, tau, is {pars[3]} with an error of {perr[3]}")
print(f"Primordial amplitude of fluctuations, As, is {pars[4]} with an error of {perr[4]}")
print(f"Slope of primordial amplitude of fluctuations, slope_ppl, is {pars[5]} with an error of {perr[5]} \n \n \n")



params_Q3=np.loadtxt("Question3_datafile.txt", delimiter=",")






burn_in_region_steps=500 # read off from the plots, should be ~ 800; but remove a bit more just in case 
params_Q3=params_Q3[burn_in_region_steps:,]

for i in range(6):
	pars[i]=np.average(params_Q3[burn_in_region_steps:,i], weights=gaussian(params_Q3[burn_in_region_steps:,3], tau_value, tau_sigma, 5000-burn_in_region_steps))
	perr[i]=np.sqrt(np.cov(params_Q3[burn_in_region_steps:,i], aweights=gaussian(params_Q3[burn_in_region_steps:,3], tau_value, tau_sigma, 5000-burn_in_region_steps)))


print("The optimized parameters with the data from Q3 and weighting with a Gaussian distribution are the following: \n")
print(f"Hubble constant, H0, is {pars[0]} with an error of {perr[0]}")
print(f"Physical baryon density, wb_h2, is {pars[1]} with an error of {perr[1]}")
print(f"Cold DM density, wc_h2, is {pars[2]} with an error of {perr[2]}")
print(f"Optical depth, tau, is {pars[3]} with an error of {perr[3]}")
print(f"Primordial amplitude of fluctuations, As, is {pars[4]} with an error of {perr[4]}")
print(f"Slope of primordial amplitude of fluctuations, slope_ppl, is {pars[5]} with an error of {perr[5]}")


