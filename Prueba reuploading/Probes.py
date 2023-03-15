#%%
from parallel_uploading import circle, density_matrix, state_labels, U_SU4
import numpy as np
import pennylane as qml

num_training = 200
num_test = 2000

Xdata, y_train = circle(num_training)
X_train = np.hstack((Xdata, np.zeros((Xdata.shape[0], 1))))

num_layers = 1
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
            qml.RY(x[0,0], wires=0)
            qml.RY(x[0,1], wires=1)
        U_SU4(p, [0,1])
    
    return qml.expval(qml.Hermitian(y, wires=[0]))


resultat = circuit(params, X_train, M)

dev._circuit.draw(output ="mpl", interactive = True)
# %%
a = [1,2]
b = [3,4]
c = a*b

print(c)
# %%
from parallel_uploading import circle, density_matrix, U_SU4
import numpy as np
import pennylane as qml

num_training = 200
num_test = 2000
num_layers = 1

params = np.random.uniform(size=(28))
label_0 = [[1], [0]]
label_1 = [[0], [1]]
state_labels = np.array([label_0, label_1])


dev = qml.device("qiskit.aer", wires = 4)
@qml.qnode(dev, interface="autograd")
def circuit(params, x, M, wires=[0,1,2,3]):

    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RX(params[2], wires=wires[2])
    qml.RX(params[3], wires=wires[3])
    
    qml.RZ(params[4], wires=wires[0])
    qml.RZ(params[5], wires=wires[1])
    qml.RZ(params[6], wires=wires[2])
    qml.RZ(params[7], wires=wires[3])

    qml.CRX(params[8], wires=[wires[3], wires[2]])
    qml.CRX(params[9], wires=[wires[3], wires[1]])
    qml.CRX(params[10], wires=[wires[3], wires[0]])

    qml.CRX(params[11], wires=[wires[2], wires[3]])
    qml.CRX(params[12], wires=[wires[2], wires[1]])
    qml.CRX(params[13], wires=[wires[2], wires[0]])

    qml.CRX(params[14], wires=[wires[1], wires[3]])
    qml.CRX(params[15], wires=[wires[1], wires[2]])
    qml.CRX(params[16], wires=[wires[1], wires[0]])

    qml.CRX(params[17], wires=[wires[0], wires[3]])
    qml.CRX(params[18], wires=[wires[0], wires[2]])
    qml.CRX(params[19], wires=[wires[0], wires[1]])

    qml.RX(params[20], wires=wires[0])
    qml.RX(params[21], wires=wires[1])
    qml.RX(params[22], wires=wires[2])
    qml.RX(params[23], wires=wires[3])

    qml.RZ(params[24], wires=wires[0])
    qml.RZ(params[25], wires=wires[1])
    qml.RZ(params[26], wires=wires[2])
    qml.RZ(params[27], wires=wires[3])

    return qml.expval(qml.Hermitian(M, wires=[0]))

M = density_matrix(state_labels[0])
resultat = circuit(params, 3, M)

dev._circuit.draw(output ="mpl", interactive = True)
# %%
import numpy as np

y_real = np.array([1,1,0,1,0,0,0,0,1,1,0,1,0,0,1,1,1,1])
y_pred = np.array([0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,1,0,0])

TP = sum(y_real * y_pred)
FP = sum(np.logical_not(y_real) * y_pred)
TN = sum(np.logical_not(y_real) * np.logical_not(y_pred))
FN = sum(y_real * np.logical_not(y_pred))

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = 2*precision*recall / (precision + recall)
specificity = TN/(TN+FP)

print(TP, FP, TN, FN)
print(accuracy, precision, recall, F1, specificity)

# %%
