import numpy as np 
import matplotlib.pyplot as plt
###### Based on the numerical derivative as shown in Question1(a) ######

def derivative(func, x, dx): 
	Newton_dx=func(x+dx)-func(x-dx)
	Newton_2dx=func(x+2*dx)-func(x-2*dx)
	numerator=8*Newton_dx-Newton_2dx
	denominator=12*dx
	return numerator/denominator

dx=np.logspace(-14, 1, 100)

##### test for e(x)

def exponential_func(x):
	return np.exp(x)

def exponential01_func(x):
	return np.exp(0.01*x)


# randomly define an x-value:
x=0

err_exp=[]
for i in dx:
	numerical_direv_exp=derivative(exponential_func, x, i)
	real_direv_exp=np.exp(x)
	difference_exp=np.abs(numerical_direv_exp-real_direv_exp)
	err_exp.append(difference_exp)

err_exp01=[]
for i in dx:
	numerical_direv_exp=derivative(exponential01_func, x, i)
	real_direv_exp=(0.01)*np.exp(0.01*x)
	difference_exp=np.abs(numerical_direv_exp-real_direv_exp)
	err_exp01.append(difference_exp)


plt.plot(dx, err_exp, '-k')
plt.xlabel("dx")
plt.ylabel("Error")
plt.title("Error size for varying step size, dx, for function f(x)=exp(x)")
plt.xscale("log")
plt.yscale("log")
plt.show()

plt.plot(dx, err_exp01, '-k')
plt.xlabel("dx")
plt.ylabel("Error")
plt.title("Error size for varying step size, dx, for function f(x)=exp(0.01x)")
plt.xscale("log")
plt.yscale("log")
plt.show()


	
optimized_dx_exp=(45*(1e-16)/4)**(1/5)

optimized_dx_exp01=(45*(1e-6)/4)**(1/5)







print(f"For an exponential function f(x)=exp(x), the optimized dx to minimize error is {optimized_dx_exp}. "
	+ "As observed from the previous graph, the value of dx at which the error is a minimum is around 0.00115."
	+ "These two results are roughly consistent.")

print(f"For an exponential function f(x)=exp(0.01x), the optimized dx to minimize error is {optimized_dx_exp01}. "
	+ "As observed from the previous graph, the value of dx at which the error is a minimum is around 0.10729."
	+ "These two results are roughly consistent.")

