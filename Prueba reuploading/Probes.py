#%%
from parallel_uploading import circle, density_matrix, state_labels, U_SU4
import numpy as np
import pennylane as qml

num_training = 200
num_test = 2000

Xdata, y_train = circle(num_training)
X_train = np.hstack((Xdata, np.zeros((Xdata.shape[0], 1))))

num_layers = 3
params = np.random.uniform(size=(num_layers + 1, 15))
M = density_matrix(state_labels[0])

dev = qml.device("qiskit.aer", wires = 2)
@qml.qnode(dev, interface="autograd")
def circuit(params, x, y):
    """A variational quantum circuit representing the Universal classifier.

    Args:
        params (array[float]): array of parameters
        x (array[float]): single input vector
        y (array[float]): single output state density matrix

    Returns:
        float: fidelity between output state and input
    """
    
    
    
    for i,p in enumerate(params):
        
        if i != 0:
            qml.Rot(x[0,0], 0, 0, wires=0)
            qml.Rot(x[0,1], 0, 0, wires=1)
        U_SU4(p, [0,1])
    
    return qml.expval(qml.Hermitian(y, wires=[0]))


print(X_train)
resultat = circuit(params, X_train, M)
print(resultat)

dev._circuit.draw(output ="mpl", interactive = True)
# %%
