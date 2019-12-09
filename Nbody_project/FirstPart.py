from functions import * 


summary = open("Summary.txt", "w")


################### Question 1 ####################
n=int(1) # numbers of particle 
grid_size = 500 
soften = 10 
mass = 1/n 
v_x = 0 # initial v in x-direction 
v_y = 0 # initial v in y-direction 

system = Particles(mass, v_x, v_y, n, grid_size, soften)
dt = 1 
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on = False, xlim = (0, grid_size), ylim = (0, grid_size))
particles_plot ,  = ax.plot([], [], "*")
def animation_plot(i):
	global system, ax, fig, dt
	system.evolve(dt)
	particles_plot.set_data(system.loc[:,0], system.loc[:,1])
	return particles_plot, 

gif = anmt.FuncAnimation(fig, animation_plot, frames = 50, interval = 5) 
gif.save("Question1.gif", writer = "imagemagick")



summary.write("################### Question 1 ##################### \n")
summary.write("As shown in the animation, the particle behaviour is stationary. \n")






