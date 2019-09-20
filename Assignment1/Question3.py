import numpy as np 

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

# testing the adaptive Simpson's rule with a few typical examples: 
test_err = 1e-8
lower_bound=-3
upper_bound=5
exp_integral, exp_count = adaptive_simpsons(np.exp, lower_bound, upper_bound, test_err)

sin_integral, sin_count = adaptive_simpsons(np.sin, lower_bound, upper_bound, test_err)

cos_integral, cos_count = adaptive_simpsons(np.cos, lower_bound, upper_bound, test_err)

def slope2(x):
	return 2*x

y_2x_integral, y_2x_count = adaptive_simpsons(slope2, lower_bound, upper_bound, test_err)

print(f"Testing for an exponential function, the integral between x={lower_bound} and "
      + f"x={upper_bound}, with an error tolerance of {test_err} gives a value of "
      + f"{exp_integral}, in {exp_count} iterations.")

print(f"Testing for a sine function, the integral between x={lower_bound} and "
      + f"x={upper_bound}, with an error tolerance of {test_err} gives a value of "
      + f"{sin_integral}, in {sin_count} iterations.")

print(f"Testing for a cosine function, the integral between x={lower_bound} and "
      + f"x={upper_bound}, with an error tolerance of {test_err} gives a value of "
      + f"{cos_integral}, in {cos_count} iterations.")

print(f"Testing for y=2x, the integral between x={lower_bound} and "
      + f"x={upper_bound}, with an error tolerance of {test_err} gives a value of "
      + f"{y_2x_integral}, in {y_2x_count} iterations.")


""" This methods calls upon the input function twice, whereas the in-class method 
calls upon the input function three times. Therefore, this method reduces calling upon 
the input function by a raito of 5/2 """