import numpy as np 
import matplotlib.pyplot as plt 


# part (a)

fulldata = np.loadtxt('229614158_PDCSAP_SC6.txt', delimiter=',')

time_fulldata=fulldata[:,0]
flux_fulldata=fulldata[:,1]

# the region to which an exponential decay will be fitted 
time_flare=time_fulldata[3200:3400]
flux_flare=flux_fulldata[3200:3400]

# defining a general exponential function (decaying one) 
def exponential_func(x, x0, b, C):
	return np.exp(-b*(x-x0))+C 


"""
By observing the plot: 
- The base line is approximately 1 instead of 0 -> C=1 
- teh amplitude is the nax flux value - the base line
"""
A=np.max(flux_flare)-1 # this gives the amplitude 
C=1 # the shift upwards by 1 due to base line being at 1 

# converting between the amplitude of an exponential function and the x0 factor 
"""
f(x)=a*np.exp(-b*(x-x0_old))+C=np.exp(-b*(x-(x0_old+ln(a)/b)))+C, where x0_new=x0_old+ln(a)/b
"""
x0_old=time_flare[0]
b=65 # vary this value by guessing and observing the behaviour of the prediction graph 
x0_new=x0_old+np.log(A)/b
flux_prediction=exponential_func(time_flare, x0_new, b, C)



plt.plot(time_flare, flux_flare, '.k', label="Original data")
plt.plot(time_flare, flux_prediction, ".r", label="Prediction")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.legend()
plt.savefig("Question2a_Flux_vs_time_at_flare.png")
plt.show()

# note that when b=65 it is somewhat of a good prediction 

print(f"The initial guess values are x0={x0_new}, b=65, C=1")
print("The model exponentially decays and is not linear.")





################################


# part (b)

# partial derivatives of the exponential function w/ respect to x0, C, b: 
def gradient_exponential_func(x, x0, b, C):
	exponential_function=np.exp(-b*(x-x0))+C 
	df_dx0=b*np.exp(-b*(x-x0))
	df_db=-(x-x0)*np.exp(-b*(x-x0))
	df_dC=1
	grad=np.zeros([x.size, 3]) # defining a zero matrix of the required size 
	grad[:, 0]=df_dx0
	grad[:, 1]=df_db
	grad[:, 2]=df_dC
	return exponential_function, grad 

# taking the code from lecture for Newton's method: 

for j in range(5):
    prediction,grad=gradient_exponential_func(time_flare, x0_new, b, C)
    r=flux_flare-prediction
    err=(r**2).sum()
    r=np.matrix(r).transpose()
    grad=np.matrix(grad)

    lhs=grad.transpose()*grad
    rhs=grad.transpose()*r
    dp=np.linalg.inv(lhs)*(rhs)
    x0_new=float(x0_new+dp[0])
    b=float(b+dp[1])
    C=float(C+dp[2])



plt.plot(time_flare, flux_flare, '.k', label="Original data")
plt.plot(time_flare, flux_prediction, ".r", label="Prediction")
plt.plot(time_flare, exponential_func(time_flare, x0_new, b, C), label="Newton's method")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.legend()
plt.savefig("Question2b_Flux_vs_time_at_flare_exponential_fit.png")
plt.show()

# comment: the guess is reasonable 


print(f"The parameters for Newton's method  are x0={x0_new}, b={b}, C={C}")




################## 
# Part (c) 
noise_matrix=np.diag(1/((flux_flare-prediction)**2))


# find errors on fit parameters 


covariance=np.linalg.inv(np.transpose(grad)@noise_matrix@grad)
error_on_fit=np.sqrt(np.diag(covariance))

print(f"The errors on the Newton's method parameters are error of {error_on_fit[0]} on {x0_new},")
print(f" error of {error_on_fit[1]} on {b},")
print(f" error of {error_on_fit[2]} on {C}. ")

# the errors are reasonably small 

##########################

# part (d)

# Definitely not. This is only limited to this one, most significant peak. 












