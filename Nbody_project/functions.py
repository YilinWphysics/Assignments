import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as anmt 





"""
Leapfrog method for higher order intergration 
notes taken from: https://rein.utsc.utoronto.ca/teaching/PSCB57_notes_lecture10.pdf
- Basic idea: stagger the updates of position and velocity in time, making the method symmetric 
"""







class Particles(): 
    def __init__(self, mass, v_x, v_y, nparticles, grid_size, soften, orbit = False): 
        """
        mass (array): mass of particle
        p_x (array): x-component of the particle's velocity  
        p_y (array): y-component of the particle's velocity 
        nparticles (int): numberof particles in the list 
        initial_condition (array): array containing initial ppsition and mass of the particles; lengths equal as nparticles 
        grid_size (int): the size of the lenght of one side of the square grid 
        soften (float): a restraint on the radius (distance of grid from origin) such that any radius smaller than soften is defined 
            to be soften, to avoid infinity when divided by in the Green function calculation 
        a (array): acceleration, [0] in x and [1] in y 
        """
        self.mass = mass 
        v_x = np.ones(nparticles) * v_x
        v_y = np.ones(nparticles) * v_y
        self.v = np.array([v_x, v_y]).transpose()
        self.nparticles = nparticles 
        self.grid_size = grid_size
        self.soften = soften
        if orbit ==False:
            self.initial_loc()
        else:
            self.initial_orbit()
        self.hist_2d()
        self.Green_func()
        self.a = np.zeros([nparticles, 2])

    def initial_loc(self):
        self.loc =  np.random.rand(self.nparticles,2) * (self.grid_size-1)
        # self.loc_y = np.random.rand(self.nparticles) * (self.grid_size-1)

    def initial_orbit(self):
        n = self.grid_size
        self.loc = np.array([[n//2+10,n//2],[n//2-10,n//2]])

    def take_one_step(self,dt):
        """"
        dt (float): size of time step 
        """
        self.loc = self.loc + self.v * dt 
        self.loc = self.loc%(self.grid_size-1)
        self.v = self.v + self.a * dt 


    def hist_2d(self):
        """
        i_loc (int): x- and y-location of particle, rounded off to nearest integer 
        grid (array): density of particle on grid of size grid_size * grid_size 
        """
        self.grid = np.zeros([self.grid_size,self.grid_size])
        self.i_loc=np.asarray(np.round(self.loc),dtype='int')
        n=self.loc.shape[0]
        for i in range(n):
            self.grid[self.i_loc[i,0],self.i_loc[i,1]]+=self.mass

    def Green_func(self): 
        """
        Motivation - calculate Green function at each grid point, 1/(4*np.pi*r) 
            where r is the distance from the origin (defined as the bottom left grid point)
        radius (float): distance from a grid point to the origin 
        Green (array): calculate Green function at each grid point 
        """
        self.Green = np.zeros([self.grid_size, self.grid_size])
        for x in range(len(self.Green[0])):
            for y in range(len(self.Green[1])):
                radius = np.sqrt(x**2 + y**2) 
                if radius < self.soften: 
                    radius = self.soften
                self.Green[x, y]=1/(4 * np.pi * radius)
        if self.grid_size%2 == 0: 
            self.Green[: self.grid_size//2, self.grid_size//2 : ] = np.flip(self.Green[: self.grid_size//2, : self.grid_size//2], axis = 1) # an intermittent step - the original grid has only been flipped once (2 x the original size)
            self.Green[ self.grid_size//2 : , :] = np.flip(self.Green[: self.grid_size//2, :], axis = 0)
        else: 
            print("Exiting - Grid size is currently odd. Pleaset set to an even value.")

    def potential(self): 
        """
        phi (array): potential obtained from the convolution, inverse Fourier transform (ifft) of Green function * ifft grid (i.e. density)
        """
        self.phi = np.real(np.fft.ifft2(np.fft.fft2(self.Green)*np.fft.fft2(self.grid)))
        self.phi = 0.5 * (np.roll(self.phi, 1, axis = 1) + self.phi)
        self.phi = 0.5 * (np.roll(self.phi, 1, axis = 0) +self.phi)

    def force(self):
        """
        force_x (array): force on each grid point in x-direction 
        force_y (array): force on each grid point in y-direction 
        """


        self.potential()
        self.force_x = -0.5 * (np.roll(self.phi, 1, axis = 0) - np.roll(self.phi, -1, axis = 0)) * self.grid
        self.force_y = -0.5 * (np.roll(self.phi, 1, axis = 1) - np.roll(self.phi, -1, axis = 1)) * self.grid

    def evolve(self, dt): 
        """
        dt: time step size from function take_one_step 
        """

        self.force()
        self.take_one_step(dt)
        for i in range(self.nparticles):
            self.a[i]= np.array([self.force_x[self.i_loc[i,0],self.i_loc[i,1]] / self.mass, self.force_y[self.i_loc[i,0],self.i_loc[i,1]]/self.mass])
        self.hist_2d()



