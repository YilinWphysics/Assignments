import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate

summary=open("Summary.txt", "w")
#Q1_saveprint=open("Q1_saveprint.txt", "w")
#Q2_saveprint=open("Q2_saveprint.txt", "w")


#################### Question 1 #############################

n=400#arbitrary, grid size is n*n
conv_criterion=1e-2 #convergence criteria 
ite=int(1e6) # maximum number of iterations 

x = np.arange(n)
y = np.arange(n)
xx, yy = np.meshgrid(x, y)
cx, cy, r = n//2, n//2, 40 # defining centre of circle to be at the centre of the grid; radius 10
condition = (xx-cx)**2+(yy-cy)**2 <=r**2


V=np.zeros([n,n])
bc=0*V

mask=np.zeros([n,n],dtype='bool')
mask[:,0]=True
mask[:,-1]=True
mask[0,:]=True
mask[-1,:]=True
mask[condition]=True 
bc[condition]=1 #arbitrarily defined potential of the cylinder 

V=bc.copy()

# one relaxation step taken on the boundary condition
b=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0 


# evolving the relaxation technique by one iteration: 
def relaxation_one_step(V, bc, mask):
    V[1:-1,1:-1]=(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0
    V[mask]=bc[mask]
    return V 

def Ax(V,mask):
    #Vuse=np.zeros([V.shape[0]+2,V.shape[1]+2])
    #Vuse[1:-1,1:-1]=V
    Vuse=V.copy()
    Vuse[mask]=0
    ans=(Vuse[1:-1,:-2]+Vuse[1:-1,2:]+Vuse[2:,1:-1]+Vuse[:-2,1:-1])/4.0
    ans=ans-V[1:-1,1:-1]
    return ans

def pad(A):
    AA=np.zeros([A.shape[0]+2,A.shape[1]+2])
    AA[1:-1,1:-1]=A
    return AA

def evolve(V, bc, mask, b, conv_criterion, ite): 
    count=0
    res=b-Ax(V,mask) #residuals 
    for i in range(ite):
        relaxation_one_step(V, bc, mask)
        res_squared=np.sum(res*res)
        res=b-Ax(V,mask)
        count=count+1
        print(f"Residual squared {res_squared}, count {count}")
        #Q1_saveprint.write(f"Residual squared {res_squared}, count {count} \n")
        if res_squared<conv_criterion: break 
    return V, count

# charge density 
rho=V[1:-1,1:-1]-(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0 



# numerically solved potential V: 
V, count = evolve(V,bc,mask,b,conv_criterion,ite) 
E = np.gradient(V[::10,::10])

# analytically solved potential V: 
def V_anl(xx, cx, yy, cy, r): 
    normalization=np.sqrt((xx.ravel()-cx)**2+(yy.ravel()-cy)**2)-r
    V=np.log(normalization)
    V=V.reshape(xx.shape)
    slope=1/V[cx, -1]
    V=-slope*V+1 
    V[condition]=1
    return V 

V_analytical=V_anl(xx, cx, yy, cy, r)
E_analytical=np.gradient(V_analytical[::10,::10])



plt.pcolormesh(V)
plt.colorbar()
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.quiver(xx[::10,::10].ravel(),yy[::10,::10].ravel(),-E[1],-E[0])
plt.title("Potential in contour, E field in arrows - Numerical")
plt.savefig("Q1_numerical.png")
plt.show()


plt.pcolormesh(rho)
plt.colorbar()
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.title("Charge density, rho")
plt.savefig("Q1_rho.png")
plt.show()

plt.pcolormesh(V_analytical)
plt.colorbar()
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.quiver(xx[::10,::10].ravel(),yy[::10,::10].ravel(),-E_analytical[1],-E_analytical[0])
plt.title("Potential in contour, E field in arrows - Analytical")
plt.savefig("Q1_analytical.png")
plt.show()

lamb=np.sum(rho)/(2*np.pi*r)

summary.write("###############Question 1################## \n")
summary.write(f"The numerical method with the relaxation technique takes {count} number of iterations to converge. \n")
summary.write(f"The line charge density lambda on the ring is {lamb}. \n")
summary.write("The numerical and analytical method yield similar results. But it should be bear in mind that the analytical method should be taken \n")
summary.write("with a grain of salt as the grid box is square and the source is circular, therefore the boundary condition of 0 cannot be properly met. ")





#################### Question 2 #############################

def evolve_conjugate_grad(V, mask, b, conv_criterion, ite, start=0): 
    count=start
    res=b-Ax(V,mask) #residuals 
    p=res.copy()
    for k in range(ite):
        Ap=(Ax(pad(p),mask))
        rtr=np.sum(res*res) # note that "rtr" here is the same concept as "res_squared" prior
        print('on iteration ' + repr(count) + ' residual is ' + repr(rtr))
        #Q2_saveprint.write(f"Residual squared {rtr}, count {count} \n")
        if rtr<conv_criterion: 
            break 
        count=count+1
        alpha=rtr/np.sum(Ap*p)
        V=V+pad(alpha*p)
        rnew=res-alpha*Ap
        beta=np.sum(rnew*rnew)/rtr
        p=rnew+beta*p
        res=rnew
    return V, count

# re-initialize: 
x = np.arange(n)
y = np.arange(n)
xx, yy = np.meshgrid(x, y)
cx, cy, r = n//2, n//2, 40 # defining centre of circle to be at the centre of the grid; radius 10
condition = (xx-cx)**2+(yy-cy)**2 <=r**2


V=np.zeros([n,n])
bc=0*V

mask=np.zeros([n,n],dtype='bool')
mask[:,0]=True
mask[:,-1]=True
mask[0,:]=True
mask[-1,:]=True

mask[condition]=True 
bc[condition]=1 #arbitrarily defined potential of the cylinder 

V=0*bc


# one relaxation step taken on the boundary condition
b=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0 

V, count = evolve_conjugate_grad(V,mask,b,conv_criterion,ite) 
E = np.gradient(V[::10,::10])

plt.pcolormesh(V)
plt.colorbar()
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.quiver(xx[::10,::10].ravel(),yy[::10,::10].ravel(),-E[1],-E[0])
plt.title("Potential in contour, E field in arrows - Numerical \n (conjugate gradient method)")
plt.savefig("Q2_numerical.png")
plt.show()


plt.pcolormesh(rho)
plt.colorbar()
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.title("Charge density, rho")
plt.savefig("Q2_rho.png")
plt.show()

plt.pcolormesh(V_analytical)
plt.colorbar()
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.quiver(xx[::10,::10].ravel(),yy[::10,::10].ravel(),-E_analytical[1],-E_analytical[0])
plt.title("Potential in contour, E field in arrows - Analytical")
plt.savefig("Q2_analytical.png")
plt.show()

summary.write("###############Question 2################## \n")
summary.write(f"The numerical method with the analytical technique takes {count} number of iterations to converge. \n")
summary.write("It is obvious that the conjugate gradient method is much faster than the relaxation method in Q1 - \n this new method takes a lot fewer steps to reach the same convergence criterion.")



#################### Question 3 #############################

def grid_interpolate (n, resolution, V, r): 
    size=np.linspace(0, n, n)
    xx, yy =np.meshgrid(size, size) # old grid
    size=np.linspace(0, n, resolution)
    xxx, yyy=np.meshgrid(size, size) # new grid (with the input resolution)
    points=np.array([xx.ravel(), yy.ravel()]).transpose()
    V=V.ravel()
    bc=np.zeros(xxx.shape)
    cx, cy, r=n//2, n//2, r
    condition=(xxx-cx)**2+(yyy-cy)**2 <=r**2
    mask=condition.copy()
    mask[:,0]=True
    mask[:,-1]=True
    mask[0,:]=True
    mask[-1,:]=True
    mask[condition]=True 
    bc[condition]=1
    V_new = scipy.interpolate.griddata(points, V, (xxx, yyy))
    return V_new, mask, bc 

def evolve_low_resolution(V, mask, b, resolution, r, n, conv_criterion, ite, multiplication_factor):

    V, count = evolve_conjugate_grad(V, mask, b, conv_criterion, ite)
    while n < resolution: 
        if n*multiplication_factor<resolution: nn=n*multiplication_factor
        else: nn = resolution # nn is the updated n 
        V, mask, bc = grid_interpolate(n, nn, V, r)
        b=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0
        V, count_add = evolve_conjugate_grad(V, mask, b, conv_criterion, ite,start=count)
        n = nn 
        count = count + count_add
    size = np.linspace(0, resolution, resolution)
    xxx, yyy = np.meshgrid(size, size)
    return V, xxx, yyy, count 




# creating a smaller resolution n and consequently defining a circle of smaller radius: 
n_low=100
r_small=10
multiplication_factor = 2

# re-initialization: 
x = np.arange(n_low)
y = np.arange(n_low)
xx, yy = np.meshgrid(x, y)
cx, cy, r = n_low//2, n_low//2, r_small # defining centre of circle to be at the centre of the grid; radius 10
condition = (xx-cx)**2+(yy-cy)**2 <=r_small**2


V=np.zeros([n_low,n_low])
bc=0*V

mask=np.zeros([n_low,n_low],dtype='bool')
mask[:,0]=True
mask[:,-1]=True
mask[0,:]=True
mask[-1,:]=True

mask[condition]=True 
bc[condition]=1 #arbitrarily defined potential of the cylinder 



V=0*bc

b=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0 

V, xxx, yyy, count = evolve_low_resolution(V, mask, b, n, r_small, n_low, conv_criterion, ite, multiplication_factor)


E=np.gradient(V[::10, ::10])


plt.pcolormesh(V)
plt.colorbar()
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.quiver(xxx[::10,::10].ravel(),yyy[::10,::10].ravel(),-E[1],-E[0])
plt.title("Potential in contour, E field in arrows")
plt.savefig("Q3_increasing_resolution.png")
plt.show()

summary.write("\n ###############Question 3################## \n")
summary.write(f"A total steps of {count} is taken to reach the same convergence criterion. \n")



#################### Question 4 #############################
 
# creating a smaller resolution n and consequently defining a circle of smaller radius: 
n_low=100
r_small=10
multiplication_factor = 2

# re-initialization: 
x = np.arange(n_low)
y = np.arange(n_low)
xx, yy = np.meshgrid(x, y)
cx, cy, r = n_low//2, n_low//2, r_small # defining centre of circle to be at the centre of the grid; radius 10
condition = (xx-cx)**2+(yy-cy)**2 <=r_small**2

cx_bump, cy_bump, r_bump = n_low//2+r_small, n_low//2, 2*r_small//10
condition_bump = (xx-cx_bump)**2+(yy-cy_bump)**2 <=r_bump**2



V=np.zeros([n_low,n_low])
bc=0*V

mask=np.zeros([n_low,n_low],dtype='bool')
mask[:,0]=True
mask[:,-1]=True
mask[0,:]=True
mask[-1,:]=True

mask[condition]=True 
bc[condition]=1 #arbitrarily defined potential of the cylinder 
mask[condition_bump]=True 
bc[condition_bump]=1 #bump on the cylinder


V=0*bc


b=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0 

V, xxx, yyy, count = evolve_low_resolution(V, mask, b, n, r_small, n_low, conv_criterion, ite, multiplication_factor)


E=np.gradient(V[::10, ::10])


plt.pcolormesh(V)
plt.colorbar()
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.quiver(xxx[::10,::10].ravel(),yyy[::10,::10].ravel(),-E[1],-E[0])
plt.title("Potential in contour, E field in arrows")
plt.savefig("Q4_10percent_bump.png")
plt.show()

summary.write("\n ###############Question 4################## \n")
summary.write(f"A total steps of {count} is taken to reach the same convergence criterion. \n")
summary.write("There should be disturbance in the field after the addition of the bump, 10 percent of the wire diameter. \n")
summary.write("However, the disturbance is not that obvious just by a quick observation with the naked eyes.\n")



#################### Question 5 #############################
def Neumann_1D(dt, dx, t_max, x_max, k, C):
    s = k*dt/dx**2
    if s > 1/2:
        print('Error - Exit')
        exit(1)
    x = np.arange(0,x_max+dx,dx) 
    t = np.arange(0,t_max+dt,dt)
    r = len(t)
    c = len(x)
    T = np.zeros([r,c])
    for n in range(0,r-1):
        for j in range(1,c-1):
            T[n,0] = t[n]*C
            T[n+1,j] = T[n,j] + s*(T[n,j-1] - 2*T[n,j] + T[n,j+1]) 
    return x,T

dt=1e-3
dx=1e-3
# to make sure the residuals don't diverge, k * dt / (dx**2) must satisfy <=0.5
k_max=0.5*dx**2/dt
k=k_max / 2 #arbitarily making sure that the chosen value of k is smaller than the maximally allowed k value 
x_max=0.01
t_max=1
C=100 
x, T=Neumann_1D(dt, dx, t_max, x_max, k, C)
time_ticks=np.arange(0, t_max/2, 0.05)

for i in time_ticks: 
    index=int(i/dt)
    plt.plot(x, T[index, :], label=f"t={i}")
plt.xlabel("Position, x")
plt.ylabel("Temperature, T")
plt.legend()
plt.savefig("Q5_heat_curves_Newmann_Robin.png")
plt.show()

summary.write("\n ###############Question 5################## \n")
summary.write("Using the 1D Neumann for heat transfer, the heat curves are plotted shown in the saved figure for Q5. \n")
summary.write("There are indeed constants that need to be set. \n")
summary.write(f"In order to make the solution converge properly, it must be that k * dt / (dx**2) <= 0.5 \n")
summary.write(f"For this calculation, these values are chosen as dt={dt}, dx={dx}, and k={k} to satisfy the aforementioned inequality.")




