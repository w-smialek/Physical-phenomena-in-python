import numpy as np
import matplotlib.pyplot as plt

###
### Parameters
###

hs = [0.6,0,0]
Js = [-1.1,-2.1,-3.8]

###

sz = np.array([[1,0],[0,-1]])
sx = np.array([[0,1],[1,0]])
one = np.eye(2)

def S_z(i):
    if i ==0:
        return np.kron(sz, np.kron(one,one))
    elif i ==1:
        return np.kron(one, np.kron(sz,one))
    elif i ==2:
        return np.kron(one, np.kron(one,sz))
    return None

def S_x(i):
    if i ==0:
        return np.kron(sx, np.kron(one,one))
    elif i ==1:
        return np.kron(one, np.kron(sx,one))
    elif i ==2:
        return np.kron(one, np.kron(one,sx))
    return None

Ham0 = (-1.0) * sum([S_x(i) for i in range(3)])

Ham1 = (-1.0) * sum([S_z(i) * hs[i] for i in range(3)]) - sum( [ Js[i+j-1] * np.matmul(S_z(i),S_z(j)) for i, j in zip([0,0,1],[1,2,2]) ] )

def Ham(l):
    return (1-l)*Ham0 + l*Ham1

def eigvecs(l):
    val, vec = np.linalg.eigh(Ham(l))

    args = np.argsort(val)
    # val = val[args]
    # vec = vec[args,:]

    return (val, vec)

def deltaE(l):
    sortevals, sortvectors = eigvecs(l)
    return sortevals[1] - sortevals[0]

def ground_state(l):
    sortevals, sortvectors = eigvecs(l)
    return sortvectors[:,0]

def mean_Sz(i,l):
    grstate = ground_state(l)
    operator = S_z(i)
    bra = grstate
    ket = np.matmul(operator, grstate)
    return np.dot(bra,ket)


xpoints = np.linspace(0,1,120)

ypoints = [deltaE(x) for x in xpoints]

plt.plot(xpoints, ypoints)
plt.xlabel(r"$\lambda$ parameter")
plt.ylabel(r"$\Delta E$", rotation="horizontal")

plt.show()

ypoints1 = [mean_Sz(0,l) for l in xpoints]
ypoints2 = [mean_Sz(1,l) for l in xpoints]
ypoints3 = [mean_Sz(2,l) for l in xpoints]


plt.plot(xpoints,ypoints1, label= r"$\langle S_1 \rangle$")
plt.plot(xpoints,ypoints2, label= r"$\langle S_2 \rangle$")
plt.plot(xpoints,ypoints3, label= r"$\langle S_3 \rangle$")
plt.legend()

plt.xlabel(r"$\lambda$ parameter")
plt.ylabel(r"$\langle S_z \rangle$", rotation="horizontal")
plt.savefig("figSzi.png", dpi =200)

yintegral = [1/a**2 for a in ypoints]

integral_val = np.trapz(yintegral, xpoints)

print(integral_val)

plt.show()