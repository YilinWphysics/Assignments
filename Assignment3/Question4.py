import numpy as np 
import matplotlib.pyplot as plt 
import camb 
from wmap_camb_example import wmap, get_spectrum


pcov = np.array([[ 1.34599766e+01,  2.27429191e-03, -2.33134798e-02,  3.88034261e-01,
   1.41048208e-09,  8.08997413e-02],
 [ 2.27429191e-03,  6.54401598e-07, -3.01065452e-06,  8.64332826e-05,
   3.33455835e-13,  1.88420375e-05],
 [-2.33134798e-02, -3.01065452e-06,  4.62480360e-05, -6.32838818e-04,
  -2.21981708e-12, -1.24799929e-04],
 [ 3.88034261e-01,  8.64332826e-05, -6.32838818e-04,  2.08038516e-02,
   7.93466765e-11,  3.08761209e-03],
 [ 1.41048208e-09,  3.33455835e-13, -2.21981708e-12,  7.93466765e-11,
   3.04274290e-19,  1.17274721e-11],
 [ 8.08997413e-02,  1.88420375e-05, -1.24799929e-04,  3.08761209e-03,
   1.17274721e-11,  6.45825601e-04]]
)

# write MCMC to fit the basic 6 parameters


length=len(wmap[:,0])

def take_step_cov(covmat):
    mychol=np.linalg.cholesky(covmat)
    return np.dot(mychol,np.random.randn(covmat.shape[0]))

tau_value=0.0544
tau_sigma=0.0073

noise = wmap[:,2]
data = wmap[:,1]
params=np.asarray([65,0.02,0.1,2e-9,0.96])
params=np.insert(params, 3, 0.05)
nstep=5000
npar=len(params)
chains=np.zeros([nstep,npar])
chisq=np.sum((data-get_spectrum(params)[2:length+2])**2/noise**2)
scale_fac=0.5
chisqvec=np.zeros(nstep)
file=open('Question4_datafile.txt', 'w') # "w" for write 
# as Prof. Sievers suggested, writing each step's results to a data file (in case of crashing)
for i in range(nstep):
    new_params=params+take_step_cov(pcov)*scale_fac
    if new_params[3]>tau_value-tau_sigma and new_params[3]<tau_value+tau_sigma:
        new_model=get_spectrum(new_params)[2:length+2]
        new_chisq=np.sum((data-new_model)**2/noise**2)
        
        delta_chisq=new_chisq-chisq
        prob=np.exp(-0.5*delta_chisq)
        accept=np.random.rand(1)<prob
        if accept:
            params=new_params
            model=new_model
            chisq=new_chisq
    chains[i,:]=params
    chisqvec[i]=chisq
    for ii in params: 
        file.write(f"{ii}, ")
    file.write(f"{chisq} \n")
    file.flush()
file.close()
    
#fit_params=np.mean(chains_new,axis=0)



