import numpy as np 
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import h5py
import glob
import os 
from scipy.constants import c 

# read file (copied frmo Prof. Siever's "simple_read_ligo.py")




def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    gpsStart=meta['GPSstart'].value
    #print meta.keys()
    utc=meta['UTCstart'].value
    duration=meta['Duration'].value
    strain=dataFile['strain']['Strain'].value
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc




directory = "LOSC_Event_tutorial"


#################### Reading the file ####################


fn_H=["H-H1_LOSC_4_V2-1126259446-32.hdf5", "H-H1_LOSC_4_V2-1128678884-32.hdf5", "H-H1_LOSC_4_V2-1135136334-32.hdf5", "H-H1_LOSC_4_V1-1167559920-32.hdf5"]
fn_L=["L-L1_LOSC_4_V2-1126259446-32.hdf5", "L-L1_LOSC_4_V2-1128678884-32.hdf5", "L-L1_LOSC_4_V2-1135136334-32.hdf5", "L-L1_LOSC_4_V1-1167559920-32.hdf5"]

fn_length=len(fn_H) # this is also the same length as len(fn_template) with fn_template assigned later 


strain_H=np.zeros([fn_length, 131072])
strain_L=np.zeros([fn_length, 131072])
# dt: time step between each data point 
dt=np.zeros([fn_length])
# utc: starting time 
utc=[]

fn_template=np.array(["GW150914_4_template.hdf5", "LVT151012_4_template.hdf5", "GW151226_4_template.hdf5", "GW170104_4_template.hdf5"])
# creating placeholder empty arrays: 
th=np.zeros([fn_length, 131072]) 
tl=np.zeros([fn_length, 131072]) 

name=["GW150914", "LVT151012", "GW151226", "GW170104"]

for i in range(fn_length): 
	strain_H[i,:], dt[i], _ = read_file(f'{directory}/{fn_H[i]}')
	strain_L[i,:], dt[i], _ = read_file(f'{directory}/{fn_L[i]}')
	th[i:,], tl[i:,] = read_template(f'{directory}/{fn_template[i]}')

#################### End of - Reading the file ####################


def noise_model(strain, window): # for part (a)
		normalization=np.sqrt(np.mean(window**2))
		noise=gaussian_filter(np.abs(np.fft.rfft(strain*window)/normalization)**2, 1)
		return noise, normalization

x=np.arange(len(strain_H[0]))
x=x-1.0*x.mean()
window=0.5*(1+np.cos(x*np.pi/np.max(x))) # use the cosine window to taper data such that the start and end are 0, compatible with fft.rfft() later 



def match_filter(template, noise, strain, window, normalization): # for part (b)
	wA = np.fft.rfft(window*template)/(np.sqrt(noise)*normalization)
	wd = np.fft.rfft(window*strain)/(np.sqrt(noise)*normalization)
	return np.fft.fftshift(np.fft.irfft(np.conj(wA)*wd))

def SNR(match_filter, template, noise, window, normalization, freq): # for part (c)
	wA = np.fft.rfft(window*template)/(np.sqrt(noise)*normalization)
	SNR = np.abs(match_filter*np.fft.fftshift(np.fft.irfft(np.sqrt(np.conj(wA)*wA)))) # for (c)
	analytic_SNR=np.abs(np.fft.irfft(wA)) # for (d)
	sum_power_spectra = np.cumsum(np.abs(wA**2)) # for (e)
	freq_half=freq[np.argmin(np.abs(sum_power_spectra-(np.amax(sum_power_spectra)/2)))]
	return SNR, analytic_SNR, freq_half

def HL_combined_SNR(SNR_H, SNR_L):
	return np.sqrt(SNR_H**2+SNR_L**2)

def gaus(x, A, sigma, mu):
	return A*np.exp(-(x-mu)**2/sigma**2)

def time_of_arrival(SNR, x):
	A=np.max(SNR)
	index=np.argmax(SNR)
	mu=x[index]
	sigma=0.0005
	params, cov = curve_fit(gaus, x[index-10:index+10], SNR[index-10:index+10], p0=[A, sigma, mu])
	return params[2], params[1]







for i in range(len(strain_H)):

	#################### Part (a) ####################
	time=np.linspace(0, len(strain_H[i])/4096, len(strain_H[i]))
	freq=np.fft.rfftfreq(len(strain_H[i]), dt[i])


	# noise model in Hanford: 
	noise_H, normalization_H=noise_model(strain_H[i], window)

	# noise model in Livingston: 
	noise_L, normalization_L=noise_model(strain_L[i], window)



	plt.subplot(2, 1, 1)
	plt.semilogy(freq, noise_H)
	plt.ylabel("Power spectrum in Hanford")
	plt.title(f'Noise model of {name[i]}')

	plt.subplot(2, 1, 2)
	plt.semilogy(freq, noise_L)
	plt.ylabel("Power spectrum in Livingston")
	plt.xlabel("Frequency (Hz)")
	plt.savefig(f"Q1a_noise_model_{name[i]}.png")
	plt.clf()



	####################  Part (b) ####################
	m_H=match_filter(th[i], noise_H, strain_H[i], window, normalization_H) # match filter for H
	m_L=match_filter(tl[i], noise_L, strain_L[i], window, normalization_L) # match filter for L


	plt.subplot(2, 1, 1)
	plt.plot(time, m_H)
	plt.ylabel("Match filter signal for Hanford")
	plt.title(f'Match filter of {name[i]}')

	plt.subplot(2, 1, 2)
	plt.plot(time, m_L)
	plt.ylabel("Match filter signal for \n Livingston")
	plt.xlabel("Time (sec)")
	plt.savefig(f"Q1b_match_filter_signal_{name[i]}.png")
	plt.clf()


	####################  Part (c) ####################
	SNR_H, analytic_SNR_H, freq_H=SNR(m_H, th[i], noise_H, window, normalization_H, freq)
	SNR_L, analytic_SNR_L, freq_L=SNR(m_L, tl[i], noise_L, window, normalization_L, freq)
	SNR_combined=HL_combined_SNR(SNR_H, SNR_L)


	plt.subplot(3, 1, 1)
	plt.plot(time, SNR_H)
	plt.ylabel("Signal-to-noise \n in Hanford")
	plt.title(f'Signal-to-noise of {name[i]}')

	plt.subplot(3, 1, 2)
	plt.plot(time, SNR_L)
	plt.ylabel("Signal-to-noise \n in Livingston")

	plt.subplot(3, 1, 3)
	plt.plot(time, SNR_combined)
	plt.ylabel("Signal-to-noise \n combined")
	plt.xlabel("Time (sec)")
	plt.savefig(f"Q1c_signal_to_noise_{name[i]}.png")
	plt.clf()

	####################  Part (d) ####################
	analytic_SNR_combined=HL_combined_SNR(SNR_H, SNR_L)


	plt.subplot(3, 1, 1)
	plt.plot(time, analytic_SNR_H)
	plt.ylabel("Analytic signal-to-noise \n in Hanford")
	plt.title(f'Analytic signal-to-noise of {name[i]}')

	plt.subplot(3, 1, 2)
	plt.plot(time, analytic_SNR_L)
	plt.ylabel("Analytic signal-to-noise \n in Livingston")


	plt.subplot(3, 1, 3)
	plt.plot(time, analytic_SNR_combined)
	plt.ylabel("Analytic_signal-to-noise \n combined")
	plt.xlabel("Time (sec)")
	plt.savefig(f"Q1d_analytic_signal_to_noise_{name[i]}.png")
	plt.clf()
	
	####################  Part (e) ####################
	print(name[i])
	print("------------")
	print(f"freq_H:{freq_H}")
	print(f"freq_L:{freq_L}") # this is saved in textfile "part_e_f_print_results.txt"

	####################  Part (f) ####################
	time_H, err_H=time_of_arrival(SNR_H, time)
	time_L, err_L=time_of_arrival(SNR_L, time)
	delta_time=np.abs(time_H-time_L)
	distance=1e6 # unit: m
	error_total=delta_time*c/distance 
	print(f"time_H:{time_H}")
	print(f"error_H:{err_H}")
	print(f"time_L:{time_L}")
	print(f"error_L:{err_L}")
	print(f"error_total:{error_total}")
	print('') # this is saved in textfile "part_e_f_print_results.txt"
