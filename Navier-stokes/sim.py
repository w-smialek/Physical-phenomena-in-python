import numpy as np
import matplotlib.pyplot as plt

###
### PARAMETERS
###

nx = 520
ny = 180
u_in = 0.04
epsilon = 0.0001
weights = [4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]
re = 1000
vlb = u_in*(ny/2)/re
tau = 3*vlb+1/2

###
### INITIAL SETUP
###
index_inverse = [0, 3, 4, 1, 2, 7, 8, 5, 6]

wedge_border = np.fromfunction(lambda i, j: abs(j-nx/4)+abs(i) == ny/2,(ny,nx)).astype(int) + np.fromfunction(lambda i, j: i==0,(ny,nx)).astype(int) + np.fromfunction(lambda i, j: i==179,(ny,nx)).astype(int)
wedge = np.fromfunction(lambda i, j: abs(j-nx/4)+abs(i) < ny/2,(ny,nx)).astype(int)

directions = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
directions_reversed = [[0, 0], [-1, 0], [0, -1], [1, 0], [0, 1], [-1, -1], [1, -1], [1, 1], [-1, 1]]

particle_distribution = np.zeros((ny,nx,9))
particle_distribution_eq = np.zeros((ny,nx,9))
velocity = np.zeros((ny,nx,9))
density = 1-wedge

velocity[:,:,1] = np.fromfunction(lambda i,j: u_in*(1+epsilon*np.sin((i*2*np.pi)/(ny-1))), (ny,nx))
velocity_init = velocity[:,:,1]

particle_distribution_eq[:,:,1] = 1/9*density[:,:]*(1+3*velocity_init+6/2*velocity_init*velocity_init)
particle_distribution = particle_distribution_eq

density = np.sum(particle_distribution,axis=2)

###
### FIRST STEP
###

# wedge_3d = np.reshape(np.array([wedge]*9),(180,520,9))
wedge_3d = np.repeat(wedge_border[:, :, np.newaxis], 9, axis=2)

particle_distribution_col = (particle_distribution - (particle_distribution - particle_distribution_eq)/tau)

particle_distribution_col_bound = particle_distribution_col[:,:,index_inverse]

particle_distribution_col = particle_distribution_col*(1-wedge_3d) + particle_distribution_col_bound*wedge_3d

plt.matshow(particle_distribution_col[:,:,1]-particle_distribution_col[:,:,3])
plt.show()

### Streaming



#particle_distribution_col + wedge_border_3d*(  )

# particle_distribution_eq[:,:,1] = 1/9*density[:,:]*(1+3*velocity_init+6/2*velocity_init*velocity_init)
# particle_distribution = particle_distribution_eq
# plt.matshow(particle_distribution_eq[:,:,1])
# plt.show()

# density = np.sum(particle_distribution,axis=2)
# density[:,0] = (2*(particle_distribution[:,0,3]+particle_distribution[:,0,6]+particle_distribution[:,0,7])+particle_distribution[:,0,0]+particle_distribution[:,0,2]+particle_distribution[:,0,4])/(1-velocity_init[:,0])

# ### Inlety / outlet


# plt.matshow(density)
# plt.show()


# # density[:,0]=