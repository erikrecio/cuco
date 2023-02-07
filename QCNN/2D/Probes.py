import pennylane as qml
import numpy as np

dev = qml.device("qiskit.aer", wires=4)

@qml.qnode(dev)
def circuit(weights):
    qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(2))
weight_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
weights = np.random.random(weight_shape)
dev._circuit.draw()