# from qiskit import QuantumRegister, ClassicalRegister
import qiskit
import numpy as np
import matplotlib.pyplot as plt

qcoin = qiskit.QuantumRegister(1,"qcoin")
ccoin = qiskit.ClassicalRegister(1,"ccoin")

circuit = qiskit.QuantumCircuit(qcoin,ccoin)

circuit.h(qcoin)
circuit.measure(qcoin,ccoin)
circuit.draw()

backend = qiskit.Aer.get_backend('qasm_simulator')
job = qiskit.execute(circuit, backend)
res = job.result()

psi = res.get_counts(circuit)
print(psi)

from qiskit.extensions import UnitaryGate

pays0 = []
pays1 = []
pays2 = []
pays3 = []
angles = np.linspace(0, np.pi/4,50)
for a in angles:
    g = a

    matrix = [[np.cos(g), 0, 0, 0-np.sin(g)*1j],
            [0, np.cos(g), 0+np.sin(g)*1j, 0],
            [0, 0+np.sin(g)*1j, np.cos(g), 0],
            [0-np.sin(g)*1j, 0, 0, np.cos(g)]]
    gate = UnitaryGate(matrix)

    matrixh = [[np.cos(g), 0, 0, 0+np.sin(g)*1j],
            [0, np.cos(g), 0-np.sin(g)*1j, 0],
            [0, 0-np.sin(g)*1j, np.cos(g), 0],
            [0+np.sin(g)*1j, 0, 0, np.cos(g)]]
    gateh = UnitaryGate(matrix)

    q = qiskit.QuantumRegister(2,"qreg")
    c = qiskit.ClassicalRegister(2,"creg")

    circuit = qiskit.QuantumCircuit(q,c)

    circuit.append(gate,q)

    circuit.p(np.pi/2,q[0])
    circuit.p(np.pi/2,q[1])

    circuit.y(q[0])
    circuit.y(q[1])

    circuit.append(gateh,q)

    circuit.measure(q,c)

    # print(circuit.draw(output='text'))

    backend = qiskit.Aer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots = 10000)
    res = job.result()

    psi = res.get_counts()
    # print(psi)

    payoff = 0
    try:
        payoff += 3*psi['00']
    except:
        pass
    try:
        payoff += 5*psi['01']
    except:
        pass
    try:
        payoff += psi['11']
    except:
        pass

    pays0.append(payoff/1024)

    q = qiskit.QuantumRegister(2,"qreg")
    c = qiskit.ClassicalRegister(2,"creg")

    circuit = qiskit.QuantumCircuit(q,c)

    circuit.append(gate,q)

    circuit.p(np.pi/2,q[0])
    circuit.p(np.pi/2,q[1])

    circuit.y(q[0])
    circuit.z(q[1])

    circuit.append(gateh,q)

    circuit.measure(q,c)

    # print(circuit.draw(output='text'))

    backend = qiskit.Aer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots = 10000)
    res = job.result()

    psi = res.get_counts()
    # print(psi)

    payoff = 0
    try:
        payoff += 3*psi['00']
    except:
        pass
    try:
        payoff += 5*psi['01']
    except:
        pass
    try:
        payoff += psi['11']
    except:
        pass

    pays1.append(payoff/1024)

    q = qiskit.QuantumRegister(2,"qreg")
    c = qiskit.ClassicalRegister(2,"creg")

    circuit = qiskit.QuantumCircuit(q,c)

    circuit.append(gate,q)

    circuit.p(np.pi/2,q[0])
    circuit.p(np.pi/2,q[1])

    circuit.z(q[0])
    circuit.y(q[1])

    circuit.append(gateh,q)

    circuit.measure(q,c)

    # print(circuit.draw(output='text'))

    backend = qiskit.Aer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots = 10000)
    res = job.result()

    psi = res.get_counts()
    # print(psi)

    payoff = 0
    try:
        payoff += 3*psi['00']
    except:
        pass
    try:
        payoff += 5*psi['01']
    except:
        pass
    try:
        payoff += psi['11']
    except:
        pass

    pays2.append(payoff/1024)

    q = qiskit.QuantumRegister(2,"qreg")
    c = qiskit.ClassicalRegister(2,"creg")

    circuit = qiskit.QuantumCircuit(q,c)

    circuit.append(gate,q)

    circuit.p(np.pi/2,q[0])
    circuit.p(np.pi/2,q[1])

    circuit.z(q[0])
    circuit.z(q[1])

    circuit.append(gateh,q)

    circuit.measure(q,c)

    # print(circuit.draw(output='text'))

    backend = qiskit.Aer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend, shots = 10000)
    res = job.result()

    psi = res.get_counts()
    # print(psi)

    payoff = 0
    try:
        payoff += 3*psi['00']
    except:
        pass
    try:
        payoff += 5*psi['01']
    except:
        pass
    try:
        payoff += psi['11']
    except:
        pass

    pays3.append(payoff/1024)

    print(g)

# print(pays)
# print(angles)
plt.plot(angles, pays0)
plt.plot(angles, pays1)
plt.plot(angles, pays2)
plt.plot(angles, pays3)

plt.savefig("fig.jpg", dpi=300)
plt.show()