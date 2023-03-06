#%%
import pennylane as qml
import numpy as np


n_qbits = 4
U = 'U_SU4'
total_params = 10
params = np.random.randn(total_params)


dev = qml.device("qiskit.aer", wires = n_qbits)
@qml.qnode(dev)
def circuit(params, U, V):
    
    return qml.probs(wires=n_qbits//2)

# print(qml.draw(circuit)(params, U, V))


circuit(params, U, V)
dev._circuit.draw(output ="mpl", interactive = True)