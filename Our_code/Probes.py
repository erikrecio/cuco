# Print circuit
#%%
import pennylane as qml
from unitary import *
import numpy as np
from global_var import *
import QCNN_circuit


print(n_qbits)
layers = int(np.log2(n_qbits))
U = 'U_SU4'
V = 'Pooling_ansatz1'
struct = "default"
total_params = dic_U_params[U]*layers + dic_U_params[V]*layers
params = np.random.randn(total_params)


dev = qml.device("qiskit.aer", wires = n_qbits)
@qml.qnode(dev)
def circuit(params, U, V):
    QCNN_circuit.QCNN_structure(U, V, params)
    return qml.probs(wires=n_qbits//2)

# print(qml.draw(circuit)(params, U, V))


circuit(params, U, V)
dev._circuit.draw(output ="mpl", interactive = True)
# %%
a = ["ojo", "Josep"]
print(a[-1])



# %%
import numpy as np

X_train, X_test = np.random.rand(20,2)*2-1,  np.random.rand(20,2)*2-1
Y_train = ((np.sum(X_train**2, axis=1) > (3/4)**2).astype(int)*2-1).tolist()
Y_test  = ((np.sum( X_test**2, axis=1) > (3/4)**2).astype(int)*2-1).tolist()
print(X_train, X_test, Y_train, Y_test)
    