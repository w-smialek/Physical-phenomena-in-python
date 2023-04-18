import numpy as np
import matplotlib.pyplot as plt
import imageio

# def RunPDGames(payoff_mat,my_strat,opp_strat,n_games):
#     result = 0
#     for i in range(n_games):
#         result += payoff_mat[my_strat,opp_strat]
#     return result

# n_games = 1

###
### Parameters
###

# b = 2.08
# payoff_mat = np.array([[1,0],[b,0]])
# grid = np.zeros((201,201)).astype(int)
# grid[100,100] = 1
# indd = np.indices((201,201))

###
###
###

# rolls = [np.roll(grid, (1-(i-i%3)//3,1-i%3), axis=(0,1)) for i in range(9)]
# # results = np.array([payoff_mat[grid,roll] for roll in rolls])
# results = np.sum(np.array([payoff_mat[grid,roll] for roll in rolls]),0)
# rolls = np.array(rolls)

# # print(results.shape)
# # print(rolls.shape)
# print("ROLLS")
# print(rolls[:,99:103,99:103])
# print("RESULTS")
# print(results[99:103,99:103])

# argss = np.argmax(results,axis=0)
# print(argss[101:102,101:102])
# plt.matshow(argss)

# indd = np.indices((201,201))
# grid_new = rolls[argss,indd[0],indd[1]]


# plt.matshow(grid_new)
# # plt.show()
# # plt.matshow(results)
# plt.show()

###
###
###

###
### Parameters
###

def RunGame(init_board,b_val,n_steps):
    grid = init_board
    indd = np.indices((201,201))
    payoff_mat = np.array([[1,0],[b_val,0]])
    for j in range(n_steps):
        rolls = [np.roll(grid, (1-(i-i%3)//3,1-i%3), axis=(0,1)) for i in range(9)]
        results = np.sum(np.array([payoff_mat[grid,roll] for roll in rolls]),0)
        results = np.array([np.roll(results, (1-(i-i%3)//3,1-i%3), axis=(0,1)) for i in range(9)])
        rolls = np.array(rolls)
        argss = np.argmax(results,axis=0)
        grid = rolls[argss,indd[0],indd[1]]
        if j%2==0:
            plt.matshow(grid)
            plt.savefig("togif"+str(j).zfill(3)+".png",dpi=300)
            plt.close()
    return grid,np.average(grid)

b = 1.9
init_grid = np.zeros((201,201)).astype(int)
init_grid[100,100] = 1

RunGame(init_grid,b,170)

filenames = ["togif"+str(aa).zfill(3)+".png" for aa in range(0,170,2)]

with imageio.get_writer('evolutiona2.gif', mode='I',duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


# random_board = np.random.randint(2,size=(201,201))
# bees = np.arange(1.0,2.2,0.03)
# effs = []

# for ind, bee in enumerate(bees):
#     current_board, cur_avg = RunGame(random_board,bee,100)
#     effs.append(cur_avg)
#     if ind%10==0:
#         plt.matshow(current_board)
#         plt.savefig("pl"+str(ind).zfill(3)+".png",dpi=300)
#         plt.close()
    
# plt.plot(bees, effs)
# plt.savefig("plot2.png",dpi=300)
# plt.show()