import numpy as np
import matplotlib.pyplot as plt

###
### PARAMETERS
###

nx = 530
ny = 190
u_in = 0.04
epsilon = 0.0001
# epsilon = 0.1
weights = [4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]
re = 1000
vlb = u_in*(ny/2)/re
tau = 3*vlb+1/2

###
### INITIAL SETUP
###


trianglel = np.fromfunction(lambda i, j: abs(j-(nx)/4)+abs(i) < (ny)/2,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: j <= (nx-10)/4,(ny,nx)).astype(int)
trianglel[trianglel > 0] = 1
trianglel[trianglel < 0] = 0

triangler = np.fromfunction(lambda i, j: abs(j-(nx)/4)+abs(i) < (ny)/2,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: j > (nx-10)/4,(ny,nx)).astype(int)
triangler[triangler > 0] = 1
triangler[triangler < 0] = 0

top = np.fromfunction(lambda i, j: i<10,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: abs(j-nx/4)+abs(i) < ny/2,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: j>nx-11,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: j<10,(ny,nx)).astype(int)
bottom = np.fromfunction(lambda i, j: i>179,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: abs(j-nx/4)+abs(i) < ny/2,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: j>nx-11,(ny,nx)).astype(int) - np.fromfunction(lambda i, j: j<10,(ny,nx)).astype(int)
top[top < 0] = 0
bottom[bottom < 0] = 0

right = np.fromfunction(lambda i, j: j>nx-11,(ny,nx)).astype(int)
left = np.fromfunction(lambda i, j: j<10,(ny,nx)).astype(int)

borders = (top+bottom+right+left+trianglel+triangler)
borders = np.repeat(borders[:,:, np.newaxis], 9, axis=2)

borders_inside = np.fromfunction(lambda i, j: i<9,(ny,nx)).astype(int) + np.fromfunction(lambda i, j: i>180,(ny,nx)).astype(int) + np.fromfunction(lambda i, j: j>nx-10,(ny,nx)).astype(int) + np.fromfunction(lambda i, j: j<9,(ny,nx)).astype(int) + np.fromfunction(lambda i, j: abs(j-(nx)/4)+abs(i+1) < (ny)/2,(ny,nx)).astype(int)
borders_inside[borders_inside < 0] = 0
borders_inside[borders_inside > 1] = 1
borders_inside = np.repeat(borders_inside[:,:, np.newaxis], 9, axis=2)

top9 = np.repeat(top[:,:, np.newaxis], 9, axis=2)
bottom9 = np.repeat(bottom[:,:, np.newaxis], 9, axis=2)
right9 = np.repeat(right[:,:, np.newaxis], 9, axis=2)
left9 = np.repeat(left[:,:, np.newaxis], 9, axis=2)
trianglel9 = np.repeat(trianglel[:,:, np.newaxis], 9, axis=2)
triangler9 = np.repeat(triangler[:,:, np.newaxis], 9, axis=2)

velocity = np.zeros((ny,nx,9))
density = np.ones((ny,nx))*(1-borders[:,:,0])
density = density.astype(float)

velocity[:,:,1] = np.fromfunction(lambda i,j: u_in*(1+epsilon*np.sin((i*2*np.pi)/(ny-1))), (ny,nx))*(1-borders[:,:,0])
inlet_velocity = velocity

# plt.matshow(velocity[triangle,:])
# plt.show()
# velocity[:,:,6] = u_in*np.fromfunction(lambda i, j: (abs(i-24)+abs(j-24))<10,(ny,nx)).astype(int)

roll_directions = [(0, 0), (0, 1), (-1, 0), (0, -1), (1, 0), (-1, 1), (-1, -1), (1, -1), (1, 1)]

def reverse_direction_triangler(a):
    b = np.zeros(np.shape(a))
    for i in [0,6,8]:
        b[:,:,i] = a[:,:,i]
    b[:,:,3] = a[:,:,3] + a[:,:,2]
    b[:,:,4] = a[:,:,4] + a[:,:,1]
    b[:,:,7] = a[:,:,7] + a[:,:,5]
    return b

def reverse_direction_trianglel(a):
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

def f_equilibrium(W, u, rho):
    f_eq = np.zeros((ny,nx,9))
    u_sq = np.sum(u*u,axis=2)#np.linalg.norm(u, axis=2)*np.linalg.norm(u, axis=2)
    for i in range(9):
        f_eq[:,:,i] = W[i]*rho*(1+3*u[:,:,i]+9/2*u[:,:,i]*u[:,:,i]-3/2*u_sq)
    return f_eq

def in_density(f, in_v):
    f_in = f[:,0,:]
    in_v_in = in_v[:,0,:]
    return (2*(f_in[:,3]+f_in[:,6]+f_in[:,7]) + f_in[:,0]+f_in[:,2]+f_in[:,4])/(1-np.sum(in_v_in,axis=1))

def in_feq(W, in_v, rho_in):
    in_v_in = in_v[:,0,:]
    u_sq = np.linalg.norm(in_v_in, axis=1)
    feq = np.zeros((190,9))
    for i in range(9):
        feq[:,i] = W[i]*rho_in*(1+3*in_v_in[:,i]+9/2*in_v_in[:,i]*in_v_in[:,i]-3/2*u_sq)
    return feq

particle_distribution_eq = f_equilibrium(weights, velocity, density)
particle_distribution = particle_distribution_eq

cxs = np.array([0, 1, 0, -1, 0, 1,-1,-1,1])
cys = np.array([0, 0, 1, 0, -1, 1,1,-1,-1])
idxs = np.arange(9)

for st in range(250):

    density = np.sum(particle_distribution,2) + 0.001
    ux  = np.sum(particle_distribution*cxs,2) / density
    uy  = np.sum(particle_distribution*cys,2) / density

    density_in = in_density(particle_distribution, inlet_velocity)

    f_in = in_feq(weights, inlet_velocity, density_in)

    for i in [1,5,8]:
        particle_distribution[:,9,i] = f_in[:,i]

    for i in [3,6,7]:
        particle_distribution[:,nx-10,i] = particle_distribution[:,nx-11,i]

    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        particle_distribution_eq[:,:,i] = w*density*(1+3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2)
    particle_distribution_eq = particle_distribution_eq*(1-borders_inside)
    collision_distribution = particle_distribution - (particle_distribution-particle_distribution_eq)*(1/tau)
    collision_distribution = collision_distribution*(1-borders_inside)

    coll_top = reverse_direction_top(collision_distribution*top9)
    coll_bottom = reverse_direction_bottom(collision_distribution*bottom9)
    # coll_right = reverse_direction_right(collision_distribution*right9)
    # coll_left = reverse_direction_left(collision_distribution*left9)
    coll_triangler = reverse_direction_triangler(collision_distribution*triangler9)
    coll_trianglel = reverse_direction_trianglel(collision_distribution*trianglel9)

    collision_distribution = collision_distribution*(1-borders) + coll_top + coll_bottom + coll_trianglel + coll_triangler

    particle_distribution = stream(collision_distribution)

    if st % 10 == 0:
        # plt.matshow(np.linalg.norm(particle_distribution,axis=2))
        plt.matshow(np.linalg.norm(particle_distribution,axis=2))
        # plt.matshow(density)
        plt.show()
        plt.close()


