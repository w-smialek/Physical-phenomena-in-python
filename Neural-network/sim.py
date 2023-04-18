import numpy as np
import matplotlib.pyplot as plt

###
### Pattern
###

pattern0 = np.array([
-1, 1, 1, 1, -1,
1, -1, -1, -1, 1,
1, -1, -1, -1, 1,
1, -1, -1, -1, 1,
1, -1, -1, -1, 1,
-1, 1, 1, 1, -1
])

pattern1 = np.array([
-1, 1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1
])

pattern2 = np.array([
1, 1, 1, -1, -1,
-1, -1, -1, 1, -1,
-1, -1, -1, 1, -1,
-1, 1, 1, -1, -1,
1, -1, -1, -1, -1,
1, 1, 1, 1, 1
])

###
### Weight matrix, Hamiltonian
###

w_mat = 1/3*(np.tensordot(pattern0,pattern0,0) + np.tensordot(pattern1,pattern1,0) + np.tensordot(pattern2,pattern2,0)) - np.identity(30)

def hamiltonian(spin_field, w):
    hamiltonian_density = -1/2*spin_field*np.matmul(w,spin_field)
    return sum(hamiltonian_density)

###
### Evolution
###

noisy_0 = np.array([
-1, 1, 1, 1, -1,
1, -1, -1, -1, -1,
1, -1, -1, -1, 1,
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, 1, -1, -1,
])

noisy_2 = np.array([
1, 1, 1, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, 1, -1, -1,
1, -1, -1, -1, -1,
1, 1, -1, -1, 1
])

noisy_2b = np.array([
1, 1, 1, -1, -1,
-1, -1, -1, 1, -1,
-1, -1, -1, 1, -1,
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1
])

evol_T = 10
evol_random_T = 150

for inputpattern in [noisy_0,noisy_2,noisy_2b]:
    energies = []
    for i in range(evol_T):
        inputpattern = np.sign(np.matmul(w_mat,inputpattern))
        energies.append(hamiltonian(inputpattern,w_mat))

    inputpattern = [[i for i in inputpattern[j:j+5]] for j in range(0,30,5)]

    plt.axes()
    plt.plot(np.arange(evol_T),energies)
    plt.show()
    plt.close()
    plt.matshow(inputpattern)
    plt.show()

###
### Random flipping
###

for inputpattern in [noisy_0,noisy_2,noisy_2b]:
    energies = []
    for i in range(evol_random_T):
        i_flip = np.random.randint(30)
        inputpattern[i_flip] = np.sign(np.matmul(w_mat,inputpattern)[i_flip])
        energies.append(hamiltonian(inputpattern,w_mat))

    inputpattern = [[i for i in inputpattern[j:j+5]] for j in range(0,30,5)]

    plt.axes()
    plt.plot(np.arange(evol_random_T),energies)
    plt.show()
    plt.close()
    plt.matshow(inputpattern)
    plt.show()