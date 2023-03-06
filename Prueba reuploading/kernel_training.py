import numpy as np
from datetime import datetime
import os.path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor

import matplotlib.pyplot as plt

np.random.seed(42)

def circle(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = -1
        if np.linalg.norm(x - center) < radius:
            y = 1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals), np.array(yvals)

def data_load_and_process(dataset, n_train, n_test, binary=True):
    X_train, X_test = np.random.rand(n_train,2)*2-1,  np.random.rand(n_test,2)*2-1

    if dataset == 'circle':
        radius = 3/4   # 1/4 or 3/4 or np.sqrt(2 / np.pi)
        Y_train = ((np.sum(X_train**2, axis=1) > radius**2).astype(int)*2-1).tolist()
        Y_test  = ((np.sum( X_test**2, axis=1) > radius**2).astype(int)*2-1).tolist()
    
    elif dataset == 'sinus1':
        Y_train = ((  X_train[:,1] > np.cos(3*X_train[:,0])  ).astype(int)*2-1).tolist()
        Y_test  = ((   X_test[:,1] > np.cos(3* X_test[:,0])  ).astype(int)*2-1).tolist()
    
    elif dataset == 'sinus2':
        Y_train = ((  X_train[:,1] > -np.sin(3*X_train[:,0]) ).astype(int)*2-1).tolist()
        Y_test  = ((   X_test[:,1] > -np.sin(3* X_test[:,0]) ).astype(int)*2-1).tolist()
    
    else:
        print("Incorrect dataset")
        return False
    
    if binary:
        Y_train = [1 if y == 1 else 0 for y in Y_train]
        Y_test = [1 if y == 1 else 0 for y in Y_test]
        
    return X_train, X_test, Y_train, Y_test


dataset = "sinus2"  # circle, sinus1, sinus2
num_training = 200
num_test = 2000

X_train, X_test, y_train, y_test = data_load_and_process(dataset, num_training, num_test, binary = False)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_qubits = len(X_train[0])

dev_kernel = qml.device("default.qubit", wires=n_qubits)

projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

@qml.qnode(dev_kernel, interface="autograd")
def kernel(x1, x2):
    """The quantum kernel."""
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))


def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])

print("SVM fit starts.")
svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
print("SVM fit ends. Predictions start.")

predictions = svm.predict(X_test)
print("Predictions end. Accuracy starts.")

acc = accuracy_score(predictions, y_test)
print("Accuracy ends.")


# Image plotting
fig = plt.figure(figsize=(11.2, 5))
fig.suptitle(f"Truth vs Predicted - Kernel training, acc = {acc}")

ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(list(zip(*X_test))[0], list(zip(*X_test))[1], 5, y_test)

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(list(zip(*X_test))[0], list(zip(*X_test))[1], 5, predictions)

file_name = f'{datetime.now().strftime("%d-%m-%Y %H-%M-%S")} - Kernel training, acc = {acc}'
plt.savefig(os.path.join(os.path.dirname(__file__), f'Plots\\{file_name}.png'))
# plt.show()








