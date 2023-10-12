import numpy as np
import matplotlib.pyplot as plt

###
###
###

def crange(start, stop, modulo):
    result = []
    index = start
    while index != stop:
        result.append(index)
        index = (index + 1) % modulo
    return result

nx = 80
ny = 80

h = 2
alpha = 9

feromone_str_decay = 0.98
feromone_evap = 0.993

nest = [nx//2,ny//2]

###
###
###

field_feromone_home = np.zeros((nx,ny)).astype(float)
field_feromone_search = np.zeros((nx,ny)).astype(float)
field_food = np.zeros((nx,ny)).astype(float)
field_food[3:20,20:60] = 3
field_nest = np.zeros((nx,ny)).astype(float)
field_nest[35:46,35:46] = 1

directions = [[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1]]

class ant:
    def __init__(self, position, direction):
        self.position = np.array(position)
        self.direction = direction
        self.iffood = False
        self.feromone_str = 1
    
    def step_search(self):
        field_feromone_search[self.position[0],self.position[1]] += self.feromone_str
        # field_feromone_search[(self.position[0]+1)%nx,(self.position[1]+1)%ny] += self.feromone_str
        # field_feromone_search[(self.position[0]-1)%nx,(self.position[1]-1)%ny] += self.feromone_str
        # field_feromone_search[(self.position[0]+1)%nx,(self.position[1]-1)%ny] += self.feromone_str
        # field_feromone_search[(self.position[0]-1)%nx,(self.position[1]+1)%ny] += self.feromone_str
        feromones_in_direction = [field_feromone_home[(self.position + directions[i_dir])[0]%nx,(self.position + directions[i_dir])[1]%ny] for i_dir in crange(self.direction-1,(self.direction+2)%8,8)]
        probabilities_in_directions = [(h + i)**alpha for i in feromones_in_direction]
        probabilities_in_directions = probabilities_in_directions/np.sum(probabilities_in_directions)
        chosen_direction = np.random.choice([(self.direction-1)%8,self.direction,(self.direction+1)%8], p=probabilities_in_directions)
        self.direction = chosen_direction
        self.position = np.remainder(self.position + directions[chosen_direction], [nx,ny])

        self.feromone_str *= feromone_str_decay


    def step_home(self):
        field_feromone_home[self.position[0],self.position[1]] += self.feromone_str
        # field_feromone_home[(self.position[0]+1)%nx,(self.position[1]+1)%ny] += self.feromone_str
        # field_feromone_home[(self.position[0]-1)%nx,(self.position[1]-1)%ny] += self.feromone_str
        # field_feromone_home[(self.position[0]+1)%nx,(self.position[1]-1)%ny] += self.feromone_str
        # field_feromone_home[(self.position[0]-1)%nx,(self.position[1]+1)%ny] += self.feromone_str
        feromones_in_direction = [field_feromone_search[(self.position + directions[i_dir])[0]%nx,(self.position + directions[i_dir])[1]%ny] for i_dir in crange(self.direction-1,(self.direction+2)%8,8)]
        # print(feromones_in_direction)
        probabilities_in_directions = [(h + i)**alpha for i in feromones_in_direction]
        probabilities_in_directions = probabilities_in_directions/np.sum(probabilities_in_directions)
        chosen_direction = np.random.choice([(self.direction-1)%8,self.direction,(self.direction+1)%8], p=probabilities_in_directions)
        self.direction = chosen_direction
        self.position = np.remainder(self.position + directions[chosen_direction], [nx,ny])

        self.feromone_str *= feromone_str_decay

    def check_for_food(self):
        pointed_point = np.remainder(self.position + directions[self.direction], [nx,ny])
        iffood_in_direction = bool(field_food[pointed_point[0],pointed_point[1]])
        if iffood_in_direction or bool(field_food[self.position[0],self.position[1]]):
            if self.iffood:
                self.direction += 4
                self.direction = self.direction%8
            else:
                self.direction += 4
                self.direction = self.direction%8
                self.iffood = True
                field_food[pointed_point[0],pointed_point[1]] += -1
                self.feromone_str = 1        


    def check_for_nest(self):
        pointed_point = np.remainder(self.position + directions[self.direction], [nx,ny])
        ifnest_in_direction = bool(field_nest[pointed_point[0],pointed_point[1]])
        if ifnest_in_direction or bool(field_nest[self.position[0],self.position[1]]):
            if self.iffood:
                # self.direction += 4
                # self.direction = self.direction%8
                self.iffood = False
                self.feromone_str = 1        
            else:
                pass
                # self.direction += 4
                # self.direction = self.direction%8
                # self.iffood = True
                # field_food[pointed_point[0],pointed_point[1]] += -1

        

ants = [ant(nest, np.random.choice(np.arange(8))) for i in range(150)]

for i in range(15000):
    ants_locations = np.zeros((nx,ny))
    ants_locations += field_food

    for a in ants:

        a.check_for_food()
        if a.iffood:
            a.step_home()
        else:
            a.step_search()
        # print(a.position)
        ants_locations[a.position[0],a.position[1]] += 1

    field_feromone_search *= feromone_evap
    field_feromone_home *= feromone_evap
    field_food[field_food<0] = 0

    field_feromone_search[nest[0],nest[1]] = 0

    if i%1000 == 0:
        plt.matshow(ants_locations)
        plt.matshow(field_feromone_search)
        plt.matshow(field_feromone_home)
        plt.show()
