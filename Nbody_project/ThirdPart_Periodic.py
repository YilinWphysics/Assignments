from functions import * 

############## Periodic B.C's ##################

summary = open("Summary.txt", "a")
Q3_periodic_E = open("Q3_periodic_E.txt", "w")


n=int(2e5) # now use hundreds of thousands of particles 
grid_size = 500
soften = 10 
mass = 1/n
v_x = 0 # initial v in x-direction 
v_y = 0 # initial v in y-direction 

system = Particles(mass, v_x, v_y, n, grid_size, soften)
dt = 80
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on = False, xlim = (0, grid_size), ylim = (0, grid_size))
particles_plot = ax.imshow(system.grid, origin='lower', vmin=system.grid.min(), vmax=system.grid.max(), cmap=plt.get_cmap('cividis'))
plt.colorbar(particles_plot)

count = 0
def animation_plot(i):
	global system, ax, fig, dt, Q3_periodic_E, count
	print(count)
	for i in range(10): 
		system.evolve(dt)
		system.energy()
		Q3_periodic_E.write(f"{system.E}")
	count+=1
	particles_plot.set_data(system.grid)
	particles_plot.set_clim(system.grid.min(), system.grid.max())
	particles_plot.set_cmap(plt.get_cmap("cividis"))
	return particles_plot,

animation_periodic = anmt.FuncAnimation(fig, animation_plot, frames = 200, interval = 10) 
animation_periodic.save("Question3_periodic.gif", writer = "imagemagick")
#plt.show()






