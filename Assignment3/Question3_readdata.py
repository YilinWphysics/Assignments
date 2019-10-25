import numpy as np 
import matplotlib.pyplot as plt 

parameters=np.loadtxt("Question3_datafile.txt", delimiter=",")
H0=parameters[:,0]
wb_h2=parameters[:,1]
wc_h2=parameters[:,2]
tau=parameters[:,3]
As=parameters[:,4]
slope_ppl=parameters[:,5]
chisq=parameters[:,6]

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
plt.show()
figure.savefig("Question3_MCMC_chain.png")

burn_in_region_steps=500 # read off from the plots, should be ~ 190; but remove a bit more just in case 
params_selected_region=parameters[burn_in_region_steps:,]
pars=np.mean(params_selected_region,axis=0)
perr=np.std(params_selected_region,axis=0)

print("The optimized parameters are the following: \n")
print(f"Hubble constant, H0, is {pars[0]} with an error of {perr[0]}")
print(f"Physical baryon density, wb_h2, is {pars[1]} with an error of {perr[1]}")
print(f"Cold DM density, wc_h2, is {pars[2]} with an error of {perr[2]}")
print(f"Optical depth, tau, is {pars[3]} with an error of {perr[3]}")
print(f"Primordial amplitude of fluctuations, As, is {pars[4]} with an error of {perr[4]}")
print(f"Slope of primordial amplitude of fluctuations, slope_ppl, is {pars[5]} with an error of {perr[5]}")


figure, (ax1, ax2, ax3, ax4, ax5, ax6, ax7)=plt.subplots(nrows=7, ncols=1, sharex=True) 
ax1.plot(np.fft.rfft(H0))
ax1.set_xscale('log')
ax1.set_yscale('symlog')
ax1.set_ylabel("H0")
ax2.plot(np.fft.rfft(wb_h2))
ax2.set_xscale('log')
ax2.set_yscale('symlog')
ax2.set_ylabel("wb_h2")
ax3.plot(np.fft.rfft(wc_h2))
ax3.set_xscale('log')
ax3.set_yscale('symlog')
ax3.set_ylabel("wc_h2")
ax4.plot(np.fft.rfft(tau))
ax4.set_xscale('log')
ax4.set_yscale('symlog')
ax4.set_ylabel("tau")
ax5.plot(np.fft.rfft(As))
ax5.set_xscale('log')
ax5.set_yscale('symlog')
ax5.set_ylabel("As")
ax6.plot(np.fft.rfft(slope_ppl))
ax6.set_xscale('log')
ax6.set_yscale('symlog')
ax6.set_ylabel("slope_ppl")
ax7.plot(np.fft.rfft(chisq))
ax7.set_xscale('log')
ax7.set_yscale('symlog')
ax7.set_ylabel("chi_squared")
ax7.set_xlabel("Steps")
plt.show()
figure.savefig("Question3_MCMC_FFT.png")

