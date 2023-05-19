import numpy as np
import matplotlib.pyplot as plt

###
###
###

nx = 100
ny = 100

beta = 0.05
rr = 10
del_h = 2
tresh = 60

X, Y = np.meshgrid(range(nx),range(ny))

elevation = Y + np.random.rand(nx,ny)*1e-6 + 100

def w(x):
    if x>=0:
        return np.exp(beta*x)
    else:
        return 0

def udlr_probabilities(coord):
    x = int(coord[1])
    y = int(coord[0])
    udlr_heights = [elevation[(y-1)%ny,x%nx],elevation[(y+1)%ny,x%nx],elevation[y%ny,(x-1)%nx],elevation[y%ny,(x+1)%nx]]
    current_height = elevation[y%ny,x%nx]
    probs = [w(current_height - h) for h in udlr_heights]
    try:
        probs = [a/sum(probs) for a in probs]
    except:
        # print(probs)
        # print(udlr_heights)
        # print(y,x)
        probs = [1/4,1/4,1/4,1/4]
    return probs

def make_avalanche(h):
    delta_h_up = np.roll(h,(-1,0),axis=(0,1)) - h
    delta_h_down = np.roll(h,(1,0),axis=(0,1)) - h
    delta_h_left = np.roll(h,(0,1),axis=(0,1)) - h
    delta_h_right = np.roll(h,(0,-1),axis=(0,1)) - h
    where_avalanche_up = np.zeros((nx,ny))
    where_avalanche_up[delta_h_up > rr] = 1
    where_avalanche_down = np.zeros((nx,ny))
    where_avalanche_down[delta_h_down > rr] = 1
    where_avalanche_left = np.zeros((nx,ny))
    where_avalanche_left[delta_h_left > rr] = 1
    where_avalanche_right = np.zeros((nx,ny))
    where_avalanche_right[delta_h_right > rr] = 1
    where_avalanche = (where_avalanche_left + where_avalanche_right + where_avalanche_up + where_avalanche_down).astype(bool)
    where_avalanche[0,:] = np.zeros(100)

    while np.any(where_avalanche):
        h[where_avalanche] += 0.25*del_h
        delta_h_up = np.roll(h,(-1,0),axis=(0,1)) - h
        delta_h_down = np.roll(h,(1,0),axis=(0,1)) - h
        delta_h_left = np.roll(h,(0,1),axis=(0,1)) - h
        delta_h_right = np.roll(h,(0,-1),axis=(0,1)) - h
        where_avalanche_up = np.zeros((nx,ny))
        where_avalanche_up[delta_h_up > rr] = 1
        where_avalanche_down = np.zeros((nx,ny))
        where_avalanche_down[delta_h_down > rr] = 1
        where_avalanche_left = np.zeros((nx,ny))
        where_avalanche_left[delta_h_left > rr] = 1
        where_avalanche_right = np.zeros((nx,ny))
        where_avalanche_right[delta_h_right > rr] = 1
        where_avalanche = (where_avalanche_left + where_avalanche_right + where_avalanche_up + where_avalanche_down).astype(bool)
        where_avalanche[0,:] = np.zeros(100)

    return h

all_river_paths = []

udlr = [[-1,0],[1,0],[0,-1],[0,1]]

for k in range(1400):
    river_path = []
    random_coord = list(np.random.randint(1,ny-1,size=2))
    river_path.append(random_coord)
    for i in range(1000):
        if river_path[-1][0] == 1 or river_path[-1][0] == 98:
            break
        current_probs = udlr_probabilities(river_path[-1])
        new_coord = [a+b for a, b in zip(udlr[np.random.choice(4,p=current_probs)],river_path[-1])]
        river_path.append(new_coord)
    
    # if k%10 == 0:
    elevation = make_avalanche(elevation)
    print(k)

    where_wet = np.zeros((nx,ny))
    for co in river_path:
        where_wet[co[0]%ny,co[1]%nx] += 1
    where_wet[where_wet>0] = 1

    elevation += -1*where_wet
    
    all_river_paths.append(river_path)


plt.imshow(np.flip(elevation,0),cmap='hot')
plt.savefig("fig1.png")
plt.show()
plt.close()
###
### Check for rivers
###

all_river_paths = []

for y in range(1,99):
    for x in range(100):
        river_path = []
        river_path.append([y,x])
        for i in range(1000):
            if river_path[-1][0] == 1 or river_path[-1][0] == 98:
                break
            current_probs = udlr_probabilities(river_path[-1])
            new_coord = [a+b for a, b in zip(udlr[np.random.choice(4,p=current_probs)],river_path[-1])]
            river_path.append(new_coord)
        all_river_paths.append(river_path)
        if x%10 ==0 and y%10 ==0:
            print(x,y)

where_wet_all = np.zeros((nx,ny))

for riv in all_river_paths:
    where_wet = np.zeros((nx,ny))
    for co in riv:
        where_wet[co[0]%ny,co[1]%nx] += 1
    where_wet[where_wet>0] = 1
    where_wet_all += where_wet

where_wet_all[where_wet_all <= tresh] = 0
where_wet_all[where_wet_all > tresh] = 1


plt.matshow(np.flipud(where_wet_all))
plt.savefig("fig2.png")
plt.show()
