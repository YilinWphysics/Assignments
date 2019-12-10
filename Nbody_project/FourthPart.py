from functions import * 
import matplotlib.colors as colors


############## Periodic B.C's ##################

summary = open("Summary.txt", "a")


n=int(2e5) # now use hundreds of thousands of particles 
grid_size = 500
soften = 10
mass = 10000
v_x = 0 # initial v in x-direction 
v_y = 0 # initial v in y-direction 



system = Particles(mass, v_x, v_y, n, grid_size, soften, early_uni = True)
grid = system.grid.copy()
grid[grid==0] = grid[grid!=0].min()*1e-3
dt = 100
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on = False, xlim = (0, grid_size), ylim = (0, grid_size))
particles_plot = ax.imshow(system.grid, origin='lower', cmap=plt.get_cmap('cividis'), norm = colors.LogNorm(vmin=grid.min(), vmax=grid.max()))
plt.colorbar(particles_plot)

count = 0
def animation_plot(i):
	global system, ax, fig, dt, count
	
	print(count)
	for i in range(5): 
		system.evolve(dt)
		print(f'step = {i}')

	count+=1
	grid = system.grid.copy()
	grid[grid==0] = grid[grid!=0].min()*1e-3
	print(system.v[0])
	print(system.a[0])
	print(system.loc[0])
	particles_plot.set_data(system.grid)
	particles_plot.set_norm(colors.LogNorm(vmin=grid.min(), vmax=grid.max()))
	particles_plot.set_cmap(plt.get_cmap("cividis"))

	return particles_plot,

animation_periodic = anmt.FuncAnimation(fig, animation_plot, frames = 100, interval = 10) 
animation_periodic.save("Question4_periodic.gif", writer = "imagemagick")
# plt.show()









