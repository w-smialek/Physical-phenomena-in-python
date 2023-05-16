import numpy as np
import matplotlib.pyplot as plt

###
### PARAMETERS
###

nx = 520
ny = 180
u_in = 0.04
# epsilon = 0.0001
epsilon = 0.1
weights = [4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]
re = 1000
vlb = u_in*(ny/2)/re
tau = 3*vlb+1/2

###
### INITIAL SETUP
###

# wedge = np.fromfunction(lambda i, j: abs(j-nx/4)+abs(i) < ny/2,(ny,nx)).astype(int) + np.fromfunction(lambda i, j: i==0,(ny,nx)).astype(int) + np.fromfunction(lambda i, j: i==179,(ny,nx)).astype(int)
# wedge[wedge > 0] = 1
# wedge[0,0] = 0
# wedge[0,nx-1] = 0
# wedge[179,0] = 0
# wedge[179,nx-1] = 0

wedge_inside = np.fromfunction(lambda i, j: abs(j-nx/4)+abs(i-1) < ny/2,(ny,nx)).astype(int)

trianglel = np.fromfunction(lambda i, j: abs(j-nx/4)+abs(i) < ny/2,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: j <= nx/4,(ny,nx)).astype(int)
trianglel[trianglel > 0] = 1
trianglel[trianglel < 0] = 0

triangler = np.fromfunction(lambda i, j: abs(j-nx/4)+abs(i) < ny/2,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: j > nx/4,(ny,nx)).astype(int)
triangler[triangler > 0] = 1
triangler[triangler < 0] = 0

top = np.fromfunction(lambda i, j: i==0,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: abs(j-nx/4)+abs(i) < ny/2,(ny,nx)).astype(int)
bottom = np.fromfunction(lambda i, j: i==179,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: abs(j-nx/4)+abs(i) < ny/2,(ny,nx)).astype(int)
top[top < 0] = 0
bottom[bottom < 0] = 0

right = np.fromfunction(lambda i, j: j==nx-1,(ny,nx)).astype(int)
left = np.fromfunction(lambda i, j: j==0,(ny,nx)).astype(int)


# plt.matshow(top)
# plt.matshow(bottom)
plt.matshow(trianglel)
plt.matshow(triangler)
# plt.matshow(triangler+trianglel+top+bottom)
plt.show()

velocity = np.zeros((ny,nx,9))
density = np.ones((ny,nx))#1-wedge_inside
density = density.astype(float)

velocity[:,:,4] = u_in*np.fromfunction(lambda i, j: (abs(i-100)+abs(j-20))<10,(ny,nx)).astype(int)
# velocity[:,:,1] = np.fromfunction(lambda i,j: u_in*(1+epsilon*np.sin(((j)*2*np.pi)/(ny-1))), (ny,nx))
velocity[:,:,3] = u_in*np.fromfunction(lambda i, j: (abs(i-20)+abs(j-20))<10,(ny,nx)).astype(int)
velocity[:,:,6] = u_in*np.fromfunction(lambda i, j: (abs(i-20)+abs(j-20))<10,(ny,nx)).astype(int)
density = np.fromfunction(lambda i, j: (abs(i-20)+abs(j-20))<10,(ny,nx)).astype(int)
inlet_velocity = velocity

plt.matshow(velocity[:,:,1])
plt.show()

def f_equilibrium(W, u, rho):
    f_eq = np.zeros((ny,nx,9))
    u_sq = np.sum(u*u,axis=2)#np.linalg.norm(u, axis=2)*np.linalg.norm(u, axis=2)
    for i in range(9):
        # f_eq[:,:,i] = W[i]*rho*(1+3*u[:,:,i]+9/2*u[:,:,i]*u[:,:,i]-3/2*u_sq)
        f_eq[:,:,i] = W[i]*rho*(3*u[:,:,i]+9/2*u[:,:,i]*u[:,:,i])#-3/2*u_sq)
    return f_eq

particle_distribution_eq = f_equilibrium(weights, velocity, density)

###
### FUNCTIONS
###

directions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
r_directions = [0, 3, 4, 1, 2, 7, 8, 5, 6]
# roll_directions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
roll_directions = [(0, 0), (0, 1), (-1, 0), (0, -1), (1, 0), (-1, 1), (-1, -1), (1, -1), (1, 1)]

def in_density(f, in_v):
    f_in = f[:,0,:]
    in_v_in = in_v[:,0,:]
    return (2*(f_in[:,3]+f_in[:,6]+f_in[:,7]) + f_in[:,0]+f_in[:,2]+f_in[:,4])/(1-np.sum(in_v_in,axis=1))

def in_feq(W, in_v, rho_in):
    in_v_in = in_v[:,0,:]
    u_sq = np.linalg.norm(in_v_in, axis=1)
    feq = np.zeros((180,9))
    for i in range(9):
        feq[:,i] = W[i]*rho_in*(1+3*in_v_in[:,i]+9/2*in_v_in[:,i]*in_v_in[:,i]-3/2*u_sq)
    return feq

def recalculate_density(f):
    return np.sum(f,axis=2)

def reverse_direction_trianglel(a):
    b = np.zeros(np.shape(a))
    for i in [0,6,8]:
        b[:,:,i] = a[:,:,i]
    b[:,:,3] = a[:,:,3] + a[:,:,2]
    b[:,:,4] = a[:,:,4] + a[:,:,1]
    b[:,:,7] = a[:,:,7] + a[:,:,5]
    return b

def reverse_direction_triangler(a):
    b = np.zeros(np.shape(a))
    for i in [0,5,7]:
        b[:,:,i] = a[:,:,i]
    b[:,:,1] = a[:,:,1] + a[:,:,2]
    b[:,:,4] = a[:,:,4] + a[:,:,3]
    b[:,:,8] = a[:,:,8] + a[:,:,6]
    return b

def reverse_direction_top(a):
    b = np.zeros(np.shape(a))
    for i in [0,1,3]:
        b[:,:,i] = a[:,:,i]
    for i in [4,7,8]:
        b[:,:,i] = a[:,:,i] + a[:,:,i-2]
    return b

def reverse_direction_bottom(a):
    b = np.zeros(np.shape(a))
    for i in [0,1,3]:
        b[:,:,i] = a[:,:,i]
    for i in [2,5,6]:
        b[:,:,i] = a[:,:,i] + a[:,:,i+2]
    return b

def reverse_direction_right(a):
    b = np.zeros(np.shape(a))
    for i in [0,2,4]:
        b[:,:,i] = a[:,:,i]
    b[:,:,3] = a[:,:,3] + a[:,:,1]
    b[:,:,6] = a[:,:,6] + a[:,:,5]
    b[:,:,7] = a[:,:,7] + a[:,:,8]
    return b

def reverse_direction_left(a):
    b = np.zeros(np.shape(a))
    for i in [0,2,4]:
        b[:,:,i] = a[:,:,i]
    b[:,:,1] = a[:,:,1] + a[:,:,3]
    b[:,:,5] = a[:,:,5] + a[:,:,6]
    b[:,:,8] = a[:,:,8] + a[:,:,7]
    return b


def funct_on_subset(funct, set, field):
    return (1-set)*field + set*funct(field)

def stream(field):
    for i in range(9):
        field[:,:,i] = np.roll(field[:,:,i],shift=roll_directions[i],axis=(0,1))
    return field

def setzero(a):
    return np.zeros(np.shape(a))

###
### STEP
###

particle_distribution = particle_distribution_eq

# plt.matshow(particle_distribution[:,:,1])
# plt.matshow(density)
# plt.show()

for st in range(250):

    # density_in = in_density(particle_distribution, inlet_velocity)

    # f_in = in_feq(weights, inlet_velocity, density_in)

    # for i in [1,5,8]:
    #     particle_distribution[:,0,i] = f_in[:,i]

    # for i in [3,6,7]:
    #     particle_distribution[:,nx-1,i] = particle_distribution[:,nx-2,i]

    density = recalculate_density(particle_distribution)
    particle_distribution_eq = f_equilibrium(weights, velocity, density)

    collision_distribution = particle_distribution# - (particle_distribution-particle_distribution_eq)/tau

    collision_distribution = funct_on_subset(reverse_direction_top, np.repeat(top[:,:, np.newaxis], 9, axis=2), collision_distribution)

    collision_distribution = funct_on_subset(reverse_direction_bottom, np.repeat(bottom[:,:, np.newaxis], 9, axis=2), collision_distribution)

    collision_distribution = funct_on_subset(reverse_direction_right, np.repeat(right[:,:, np.newaxis], 9, axis=2), collision_distribution)

    collision_distribution = funct_on_subset(reverse_direction_left, np.repeat(left[:,:, np.newaxis], 9, axis=2), collision_distribution)

    collision_distribution = funct_on_subset(reverse_direction_trianglel, np.repeat(trianglel[:,:, np.newaxis], 9, axis=2), collision_distribution)

    collision_distribution = funct_on_subset(reverse_direction_triangler, np.repeat(triangler[:,:, np.newaxis], 9, axis=2), collision_distribution)

    particle_distribution = stream(collision_distribution)#funct_on_subset(stream, np.repeat((1-wedge_inside)[:,:, np.newaxis], 9, axis=2), collision_distribution)

    # density = funct_on_subset(setzero,wedge_inside,density)
    # particle_distribution = funct_on_subset(setzero, np.repeat(wedge[:,:, np.newaxis], 9, axis=2), particle_distribution)

    # plt.matshow(np.linalg.norm(particle_distribution,axis=2))
    # # plt.matshow(density)
    # plt.show()


    for i in range(9):
        velocity[:,:,i] = particle_distribution[:,:,i]#/density

    # plt.matshow(particle_distribution[:,:,1])
    # plt.matshow(particle_distribution[:,:,6])

    if st % 12 == 0:
        plt.matshow(np.linalg.norm(particle_distribution,axis=2))
        # plt.matshow(density)
        plt.savefig("plot"+str(st)+".png")

### DONT STREAM FROM WEDGE!! - FIX THAT