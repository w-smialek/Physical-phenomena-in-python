import numpy as np
import matplotlib.pyplot as plt
import imageio

###
### PARAMETERS
###

nx = 520
ny = 180
u_in = 0.07
epsilon = 0.001
weights = [4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]
re = 350
vlb = u_in*(ny/2)/re
tau = 3*vlb+1/2

###
### INITIAL SETUP
###

roll_directions = [(0, 0), (0, 1), (-1, 0), (0, -1), (1, 0), (-1, 1), (-1, -1), (1, -1), (1, 1)]
cxs = np.array([0, 1, 0, -1, 0, 1,-1,-1,1])
cys = np.array([0, 0, 1, 0, -1, 1,1,-1,-1])
idxs = np.arange(9)

def stream(field):
    for i in range(9):
        field[:,:,i] = np.roll(field[:,:,i],shift=roll_directions[i],axis=(0,1))
    return field


X, Y = np.meshgrid(range(nx),range(ny))

f_t = np.ones((ny,nx,9))
f_t[:,:,1] +=  np.fromfunction(lambda i,j: u_in*(1+epsilon*np.sin((i*2*np.pi)/(ny-1))), (ny,nx))
f_inlet = f_t
f_eq = np.zeros(f_t.shape)

density = np.ones((ny,nx))

# ux = np.fromfunction(lambda i,j: u_in*(1+epsilon*np.sin((i*2*np.pi)/(ny-1))), (ny,nx))[1:179,:]
ux_init = np.fromfunction(lambda i,j: u_in*(1+epsilon*np.sin((i*2*np.pi)/(ny-1))), (ny,nx))[1:179,0]

# for i, cx, cy, w in zip(idxs, cxs, cys, weights):
#     f_t[1:179,:,i] = w*density[1:179,:]*(1+3*(cx*ux) + 9*(cx*ux)**2/2 - 3*(ux**2)/2)

X, Y = np.meshgrid(range(nx), range(ny))
triangle = abs(X-(nx)/4)+abs(Y) < (ny)/2
top = Y==0
bottom = Y==179
borders = triangle + top + bottom

# borders = (X - nx/4)**2 + (Y - ny/2)**2 < (ny/4)**2

# inlet od 5 do 174 włącznie

nt = 5000

for st in range(nt):
    f_t = stream(f_t)

    in_f_eq = f_t[1:179,0,:]
    in_density = (2*(in_f_eq[:,3] + in_f_eq[:,6]+in_f_eq[:,7]) + in_f_eq[:,0]+in_f_eq[:,2]+in_f_eq[:,4])*1/(1-abs(ux_init))

    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        in_f_eq[:,i] = w*in_density*(1+3*(cx*ux_init) + 9*(cx*ux_init)**2/2 - 3*(ux_init**2)/2)

    for i in [1,5,8]:
        f_t[1:179,0,i] = in_f_eq[:,i]
    # for i in [0,2,3,4,6,7]:
    #     f_t[:,0,i] = np.zeros(f_t[:,0,i].shape)

    for i in [3,6,7]:
        f_t[1:179,nx-2,i] = f_t[1:179,nx-1,i]
    # for i in [0,1,2,4,5,8]:
    #     f_t[:,nx-1,i] = np.zeros(f_t[:,nx-1,i].shape)



    f_t_borders = f_t[borders,:]
    f_t_borders = f_t_borders[:,[0, 3, 4, 1, 2, 7, 8, 5, 6]]

    density = np.sum(f_t,2)

    ux  = np.sum(f_t*cxs,2) / density
    uy  = np.sum(f_t*cys,2) / density

    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        f_eq[:,:,i] = w*density*(1+3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2)

    f_t += -(1.0/tau)*(f_t - f_eq)

    f_t[borders,:] = f_t_borders

    
    if st % 50 == 0:
        # plt.matshow(np.linalg.norm(f_t,axis=2))
        plt.imshow(np.linalg.norm(f_t,axis=2),cmap='hot')
        plt.savefig("plot"+str(st)+".png")
        print(st)
        # plt.show()
        plt.close()

filenames = ["plot"+str(st)+".png" for st in range(0,nt,50)]

with imageio.get_writer('vortex.gif', mode='I',duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
