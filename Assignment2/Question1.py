import numpy as np 
import os

from matplotlib import pyplot as plt 

x=np.linspace(-1, 1, 10000)

def chebyshev_matrix(x, n): 
	# note that n is the index of the polynomials, starting from n=0 
	matrix=np.zeros([len(x), n+1]) # creating a zero matrix of dimensions length(x) times order+1
	
	matrix[:, 0]=1 # First order of Chebyshev polynomials: T0=1
	
	if n>=1: 
		# Second order of Cheb poly: T1=x
		matrix[:,1]=x
	if n>=2: 
		# third order of Cheb poly: use recursive relation for T(n+1) by recalling T(n) and T(n-1)
		# Recall recursive relation: Tn+1(x)=2*x*Tn(x)-Tn-1(x)
		for i in range(1, n): 
			matrix[:, i+1]=2*x*matrix[:,i]-matrix[:,i-1]
	return matrix



matrix_order_30=chebyshev_matrix(x, 30)




#### mapping the interval in question, i.e. 0.5 < x < 1, to the interval accepted in the Cheb poly, i.e. -1 < x < 1: 
# simple method adapted from Stackoverflow: https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another

def mapfromto(x, a, b, c, d): 
	""" 
	x - input value 
	a, b - input range 
	c, d - output range 
	y - retunr value
	"""
	y=(x-a)/(b-a)*(d-c)+c
	return y 
# note that the old interval is a = -1, 1
x_new_interval=mapfromto(x, -1, 1, 0.5, 1)


# Compute the chebyshev polynomials for the array of x (from -1 to 1) as previously assigned, to order of 30 (arbitrary)





# The function in questoin is the log base 2 of x
# input x as the new interval as mapped onto the range of from 0.5 to 1 
func=np.log2(x_new_interval)





# Coefficients to the Cheb poly (note that this is an array of len(x): 
def chebyshev_coeffs_leastsq(matrix, func): 
	lhs=np.dot(matrix.transpose(), matrix)
	rhs=np.dot(matrix.transpose(), func)
	coeffs = np.dot(np.linalg.inv(lhs), rhs)
	return coeffs
# find the coefficients to the matrix and function in question: 
coeffs_30=chebyshev_coeffs_leastsq(matrix_order_30, func)


def truncated_cheb(x, func, n,  tolerance):

	matrix = chebyshev_matrix(x,n)
	coeffs = chebyshev_coeffs_leastsq(matrix, func)
	def error_tolerance(coeffs, tolerance): 
		max_order=len(coeffs)
		for i in range(len(coeffs)): 
			if np.abs(coeffs[i]) <= tolerance: 
				max_order=i
				break 
		return max_order
	max_index = error_tolerance(coeffs, tolerance)
	fit_func = np.dot(matrix[:,:max_index], coeffs[:max_index])
	return fit_func, max_index


fit_func, max_index=truncated_cheb(x, func, 30, 1e-6) 
# note that n=30 is an arbitrarily large value; the maximum index is later truncated using error_tolerance, as determined by the size of the tolenrance, here is 1e-6
plt.plot(x_new_interval, fit_func, '.r', label="Chebyshev fit")
plt.plot(x_new_interval, func, '-k', label="Original function")
plt.xlabel("x interval")
plt.ylabel("Functions")
plt.title("Comparison of the original function of log base 2 of x and \n the truncated Chebyshev polynomial expression of the function")
plt.legend()
plt.savefig("Question1a_compare_original_func_and_truncated_Cheb_poly.png")
plt.show()

print("The number of terms needed for an error tolerance of 1e-6 is " + str(max_index)) # prints the number of terms needed to achieve an input error tolerance 




########### This is the linear combination of the polyfit method using the numpy.polyfit method;
# the residuals are compared to that of the truncated Cheb poly method that I previously coded 


fit_linear = np.polyfit(x_new_interval, func, max_index-1)
fit_linear = np.polyval(fit_linear, x_new_interval)

plt.plot(x_new_interval, fit_func-func, '.b', label="Chebyshev method") # residuals from the Chebyshev polynomial expression 
plt.plot(x_new_interval, fit_linear-func, '.g', label="Numpy.polyfit method") # residuals from the numpy.polyfit method 
plt.xlabel("x interval")
plt.ylabel("Residuals")
plt.title("Comparison of residuals from the truncated Chebyshev method and \n numpy.polyfit method")
plt.legend()
plt.savefig("Question1a_compare_truncated_Cheb_poly_and_numpy_polyfit_methods.png")
plt.show()

print('rms error for numpy.polyfit method is ',np.sqrt(np.mean((fit_linear-func)**2)),' with max error ',np.max(np.abs(fit_linear-func)))
print('rms error for truncated Chebyshev method is ',np.sqrt(np.mean((fit_func-func)**2)),' with max error ',np.max(np.abs(fit_func-func)))

print("Comment on the comparisons on residuals: The truncated Cherbyshev method has a higher root mean square (as indicated by the larger average error), but has a smaller maximum error;")
print("whereas the numpy.polyfit method has a smaller root mean square (as indicated by the smaller average error), but has a much larger maximum error.")






'''
# Part (b): 
The way the functions are coded answers for part b already. To extend the function to take the log base 2 of any positive numnber, simply alter the x_new_interval:
Current the x range takes the value between 0.5 and 1 as asked in part (a) - this is achieved by mapping the range of -1 to 1 of the Chebyshev to the specific range of 0.5 to 1 for part (a).
To change the mapped x region, sipmly change the last two parameters in mapfromto. 
For example, to take the positive values from 3 to 10 instead of from 0.5 to 1, replace the current x_new_interval=mapfromto(x, -1, 1, 0.5, 1) with x_new_interval=mapfromto(x, -1, 1, 3, 10)


'''







