from scipy.interpolate import interp1d
from scipy.misc import derivative 
import numpy as np 
import matplotlib.pyplot as plt 

lakeshore=np.genfromtxt("lakeshore.csv")
T_data=lakeshore[:,0] #K 
V_data=lakeshore[:,1] #V
dV_dT_data=lakeshore[:,2] #mV/K

#reverse the arrays such that data is in increasing order, to suit interp1d 
T=T_data[::-1] 
V=V_data[::-1]
dV_dT=dV_dT_data[::-1]

f = interp1d(V, T, kind="cubic")

ticks=np.linspace(V[0], V[-1], 10000)

plt.plot(ticks, f(ticks), '-r', label="Interpolated data")
plt.plot(V, T, '.k', label="Empirical data")
plt.xlabel("Voltage, V (V)")
plt.ylabel("Temperature, T (K)")
plt.title("Interolation of Lakeshore diode temperature versus voltage data")
plt.legend()
plt.show()

# Estimate error: 
# based on the results from 1(a) 
def first_direv(func, x, dx):
	return (4/3)*(func(x+dx)-func(x-dx))/(2*dx)-(1/3)*(func(x+2*dx)-func(x-2*dx))/(4*dx)

err_diff=first_direv(f, V[1:-1], 1e-4) - 1/(dV_dT[1:-1]*0.001)
abs_err_diff=np.abs(err_diff)
max_abs_err_diff=np.max(abs_err_diff)
min_abs_err_diff=np.min(abs_err_diff)

# This is the error on my derivatives: 
print(abs_err_diff)

print ("The range of error on the derivative is from a minimum of " + str(min_abs_err_diff) + " to a maximum of " + str(max_abs_err_diff))