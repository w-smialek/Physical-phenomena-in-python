import numpy as np
import matplotlib.pyplot as plt
import imageio

nx = 200
ny = 200
time = 150

plot_interval = 3

repr_time = 3
energy = 5
repr_time_s = 20

n_fish = 3000
n_shark = 200

p_fish = n_fish/(nx*ny)
p_shark = n_shark/(nx*ny)

def neighbours(tup, field):
    all_neis =  [((tup[0]-1)%nx,tup[1]%ny),((tup[0]+1)%nx,tup[1]%ny),(tup[0]%nx,(tup[1]-1)%ny),(tup[0]%nx,(tup[1]+1)%ny)]
    empty_neis = []
    for nei in all_neis:
        if field[nei[0]%nx,nei[1]%ny] == 0:
            empty_neis.append(nei)
    return empty_neis

def sh_neighbours(tup, ffield,sfield):
    all_neis =  [(tup[0]-1,tup[1]),(tup[0]+1,tup[1]),(tup[0],tup[1]-1),(tup[0],tup[1]+1)]
    final_neis = []
    for nei in all_neis:
        if ffield[nei[0]%nx,nei[1]%ny] > 0:
            final_neis.append(nei)
        eating = True
    if not final_neis:
        for nei in all_neis:
            if sfield[nei[0]%nx,nei[1]%ny] == 0:
                final_neis.append(nei)
        eating = False
    return (final_neis, eating)

def image(fishes,sharkses):
    im = np.zeros((nx,ny,3))
    im[:,:,0] = 0
    im[:,:,1] = 0
    im[:,:,2] = 1

    im[sharkses>0,:] = 0
    im[fishes>0,0] = 0
    im[fishes>0,1] = 1
    im[fishes>0,2] = 0
    return im



fish_age = np.random.choice([1, 0], size=(nx,ny), p=[p_fish, 1-p_fish])

# fish_age = np.zeros((nx,ny))
# fish_age[10,10] = 1
# fish_age[10,11] = 1
# fish_age[11,10] = 1

sharks_age = np.random.choice([1, 0], size=(nx,ny), p=[p_shark, 1-p_shark])
sharks_age[fish_age>0] = 0
sharks_energy = energy * sharks_age

# plt.matshow(fish_age + 4*sharks_age + sharks_energy)
# plt.show()

fish_locs = np.transpose(np.nonzero(fish_age))

fish_data = []
shark_data = []

for i in range(time):
    print(i)
    fish_locs = np.transpose(np.nonzero(fish_age))
    fish_data.append(fish_locs.shape[0])
    for flo in fish_locs:
        neis = neighbours(flo,fish_age+sharks_age)
        try:
            new_loc = neis[np.random.choice(len(neis))]
            current_age = fish_age[flo[0]%nx,flo[1]%ny]
            if current_age%repr_time == 0:
                fish_age[flo[0]%nx,flo[1]%ny] = np.random.choice([a for a in range(repr_time)])
            else:
                fish_age[flo[0]%nx,flo[1]%ny] = 0
            fish_age[new_loc[0]%nx,new_loc[1]%ny] = current_age + 1
        except:
            fish_age[flo[0]%nx,flo[1]%ny] += 1

    shark_locs = np.transpose(np.nonzero(sharks_age))
    shark_data.append(shark_locs.shape[0])
    for shlo in shark_locs:
        neis, will_eat = sh_neighbours(shlo,fish_age,sharks_age)
        if sharks_energy[shlo[0]%nx,shlo[1]%ny] < 1:
            sharks_age[shlo[0]%nx,shlo[1]%ny] = 0
            continue
        try:
            new_loc = neis[np.random.choice(len(neis))]
            current_age = sharks_age[shlo[0]%nx,shlo[1]%ny]
            current_energy = sharks_energy[shlo[0]%nx,shlo[1]%ny]

            if current_age%repr_time_s == 0:
                sharks_age[shlo[0]%nx,shlo[1]%ny] = np.random.choice([a for a in range(repr_time_s)])
                sharks_energy[shlo[0]%nx,shlo[1]%ny] = energy
            else:
                sharks_age[shlo[0]%nx,shlo[1]%ny] = 0
                sharks_energy[shlo[0]%nx,shlo[1]%ny] = 0
            sharks_age[new_loc[0]%nx,new_loc[1]%ny] = current_age + 1
            sharks_energy[new_loc[0]%nx,new_loc[1]%ny] = current_energy - 1

            if will_eat:
                fish_age[new_loc[0]%nx,new_loc[1]%ny] = 0
                sharks_energy[new_loc[0]%nx,new_loc[1]%ny] = energy
        except:
            sharks_age[shlo[0]%nx,shlo[1]%ny] += 1
            sharks_energy[shlo[0]%nx,shlo[1]%ny] += -1

    # plt.matshow(fish_age)
    if i%plot_interval==0:
        plt.imshow(image(fish_age,sharks_age))
        plt.savefig("./Lotka-Volterra/imfish200_"+str(i).zfill(3)+".png", dpi = 200)
        plt.close()
    # plt.show()

filenames = ["./Lotka-Volterra/imfish200_"+str(i).zfill(3)+".png" for i in range(0,time,plot_interval)]

with imageio.get_writer('rybyyy2_200.gif', mode='I',duration=0.1*plot_interval) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

plt.plot(np.arange(len(shark_data)),6*np.array(shark_data))
plt.plot(np.arange(len(fish_data)),fish_data)
plt.savefig("plotnofish2_200.png")
plt.show()
plt.close()

plt.plot(fish_data, 6*np.array(shark_data))
plt.savefig("fishshark_200.png")
plt.show()