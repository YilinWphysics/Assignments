import numpy as np 
from scipy import integrate 
import matplotlib.pyplot as plt

def integrand(z, R, u): 
	numerator=z-R*u
	denominator=(R**2+z**2-2*R*z*u)**(3/2)
	return numerator/denominator

###### This is the adaptive Simpson's rule from Question 3 ######
def simpsons_rule(func, a, func_a, b, func_b): 
	# evaluating integral between x=a and x=b using Simpson's rule
	# h is the midpoint between a and b 
	h = (a+b)/2
	func_h=func(h)
	simpson_integral=(np.abs(b-a)/6) * (func_a+4*func_h+func_b)
	return h, func_h, simpson_integral

def recursive_simpson(func, a, func_a, b, func_b, err, total_int, h, func_h, count=0): 
	# Evaluating the left part (from a to h) of the integral using Simpson's rule: 
	left_h, left_func_h, left_int=simpsons_rule(func, a, func_a, h, func_h)
	# Evaluating the right part (from h to b) of the integral using Simpson's rule: 
	right_h, right_func_h, right_int=simpsons_rule(func, h, func_h, b, func_b)
	delta=left_int+right_int-total_int
	# criterion for determining when to stop subdividing an interval - when delta < 15*err 
	if np.abs(delta)<=15*err:
		return total_int, count
	else: 
		return recursive_simpson(func, a, func_a, h, func_h, err/2, left_int, left_h, left_func_h, count+1)+recursive_simpson(func, h, func_h, b, func_b, err/2, right_int, right_h, right_func_h, count+1)

def adaptive_simpsons(func, a, b, err):
	func_a, func_b = func(a), func(b)
	h, func_h, total_int = simpsons_rule(func, a, func_a, b, func_b)
	integral = recursive_simpson(func, a, func_a, b, func_b, err, total_int, h, func_h)
	count = np.sum(integral[1::2])
	integral = np.sum(integral[0::2])
	return integral, count

	##########################################

z_ticks=np.linspace(0,10,101)


########## Integrating using the quad method ##########

# Assign an arbitrary value of radius of sphere, r=1
R=1
# Lower bound of integration: 
a=-1
# Upper bound of integration: 
b=1

# Creating integral_list as an empty array for now to save values into it for plotting purpose later 
integral_quad_list=[]

for i in z_ticks:
	func = lambda x: integrand(i, R, x)
	# scipy.integrate.quad: input func (function); a (float) - lower bound; b (float) - upper bound
	# returns: y (float) - integral from a to b; abserr (float) - an estimate of the absolute error 
	integral, error = integrate.quad(func, a, b)
	integral_quad_list.append(integral)

# Plot integral_list again z_ticks: 
plt.plot(z_ticks, integral_quad_list, '-k')
plt.xlabel("z-axis")
plt.ylabel("Electric field")
plt.title("Electric field as calculated using quad integration")
plt.show()

# Comment on the plot previously plotted: Z<R is the region of E=0, and Z>R is the region where E decays as z increases; 
# There IS singularity - at z=R. However, the quad integration method does not care about the singularity. 

########## Integrating using the adaptive Simpson's rule as written in Question 3 ##########

# randomly define a tolerance level: 
err=1e-8

integral_AS_list=[] # AS: adaptive Simpson's 

for i in z_ticks: 
	func = lambda x: integrand(i, R, x)
	integral, count = adaptive_simpsons(func, a, b, err)
	integral_AS_list.append(integral)

# Note that the adaptive Simpson's rule crashes due to divergence at z=1 




