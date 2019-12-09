from functions import * 


summary = open("Summary.txt", "a")


################### Question 1 ####################
n=int(2) # numbers of particle 
grid_size = 500 
soften = 10 
mass = 5
v_x = [0,0] # initial v in x-direction 
v_y = [0.1,-0.1] # initial v in y-direction 

system = Particles(mass, v_x, v_y, n, grid_size, soften, orbit=True)
dt = 5
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on = False, xlim = (0, grid_size), ylim = (0, grid_size))
particles_plot ,  = ax.plot([], [], "*")
def animation_plot(i):
	global system, ax, fig, dt
	system.evolve(dt)
	particles_plot.set_data(system.loc[:,0], system.loc[:,1])
	return particles_plot, 

gif = anmt.FuncAnimation(fig, animation_plot, frames = 500, interval = 10) 
gif.save("Question2.gif", writer = "imagemagick")








