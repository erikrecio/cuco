# Inspirtation from https://pennylane.ai/qml/demos/tutorial_data_reuploading_classifier.html

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer

import matplotlib.pyplot as plt
from datetime import datetime
import os.path

# Set a random seed
np.random.seed(42)

# Make a dataset of points inside and outside of a circle
def circle(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    """
    Generates a dataset of points with 1/0 labels inside a given radius.

    Args:
        samples (int): number of samples to generate
        center (tuple): center of the circle
        radius (float: radius of the circle

    Returns:
        Xvals (array[tuple]): coordinates of points
        yvals (array[int]): classification labels
    """
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0
        if np.linalg.norm(x - center) < radius:
            y = 1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)


def plot_data(x, y, fig=None, ax=None):
    """
    Plot data with red/blue values for a binary classification.

    Args:
        x (array[tuple]): array of data points as tuples
        y (array[int]): array of data points as tuples
    """
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    reds = y == 0
    blues = y == 1
    ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k")
    ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


# Xdata, ydata = circle(2000)
# fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# plot_data(Xdata, ydata, fig=fig, ax=ax)
# plt.show()

# Define output labels as quantum state vectors
def density_matrix(state):
    """Calculates the density matrix representation of a state.

    Args:
        state (array[complex]): array representing a quantum state vector

    Returns:
        dm: (array[complex]): array representing the density matrix
    """
    return state * np.conj(state).T

label_0 = [[1], [0]]
label_1 = [[0], [1]]
state_labels = np.array([label_0, label_1], requires_grad=False)
# print(state_labels)


dev = qml.device("default.qubit", wires=2)
# Install any pennylane-plugin to run on some particular backend


def U_SU4(params, wires): # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])

@qml.qnode(dev, interface="autograd")
def qcircuit(params, x, y):
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
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)
        U_SU4(p, [0,1])
    
    return qml.expval(qml.Hermitian(y, wires=[0]))

def cost(params, x, y, state_labels=None):
    """Cost function to be minimized.

    Args:
        params (array[float]): array of parameters
        x (array[float]): 2-d array of input vectors
        y (array[float]): 1-d array of targets
        state_labels (array[float]): array of state representations for labels

    Returns:
        float: loss value to be minimized
    """
    # Compute prediction for each input in data batch
    loss = 0.0
    dm_labels = [density_matrix(s) for s in state_labels]
    # print(dm_labels) #!------------------------------------------------------------------------ Mirar si surt 3D: no, surt bé --------------------------
    for i in range(len(x)):
        f = qcircuit(params, x[i], dm_labels[y[i]])
        loss = loss + (1 - f) ** 2  #!-------------------------------------------------------- usem cross entropy? ------------------------
    return loss / len(x)


def test(params, x, y, state_labels=None):
    """
    Tests on a given set of data.

    Args:
        params (array[float]): array of parameters
        x (array[float]): 2-d array of input vectors
        y (array[float]): 1-d array of targets
        state_labels (array[float]): 1-d array of state representations for labels

    Returns:
        predicted (array([int]): predicted labels for test data
        output_states (array[float]): output quantum states from the circuit
    """
    fidelity_values = []
    dm_labels = [density_matrix(s) for s in state_labels]
    predicted = []

    for i in range(len(x)):
        fidel_function = lambda y: qcircuit(params, x[i], y)
        fidelities = [fidel_function(dm) for dm in dm_labels]
        best_fidel = np.argmax(fidelities)

        predicted.append(best_fidel)
        fidelity_values.append(fidelities)

    return np.array(predicted), np.array(fidelity_values)


def accuracy_score(y_true, y_pred):
    """Accuracy score.

    Args:
        y_true (array[float]): 1-d array of targets
        y_predicted (array[float]): 1-d array of predictions
        state_labels (array[float]): 1-d array of state representations for labels

    Returns:
        score (float): the fraction of correctly classified samples
    """
    score = y_true == y_pred
    return score.sum() / len(y_true)


def iterate_minibatches(inputs, targets, batch_size):
    """
    A generator for batches of the input data

    Args:
        inputs (array[float]): input data
        targets (array[float]): targets

    Returns:
        inputs (array[float]): one batch of input data of length `batch_size`
        targets (array[float]): one batch of targets of length `batch_size`
    """
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]


if __name__ == "__main__" :

    # Generate training and test data
    num_training = 200
    num_test = 2000

    Xdata, y_train = circle(num_training)
    X_train = np.hstack((Xdata, np.zeros((Xdata.shape[0], 1), requires_grad=False)))

    Xtest, y_test = circle(num_test)
    X_test = np.hstack((Xtest, np.zeros((Xtest.shape[0], 1), requires_grad=False)))


    # Train using Adam optimizer and evaluate the classifier
    num_layers = 1
    learning_rate = 0.6
    epochs = 10
    batch_size = 32

    opt = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

    # initialize random weights
    params = np.random.uniform(size=(num_layers + 1, 15), requires_grad=True)

    # predicted_train, fidel_train = test(params, X_train, y_train, state_labels)
    # accuracy_train = accuracy_score(y_train, predicted_train)

    # predicted_test, fidel_test = test(params, X_test, y_test, state_labels)
    # accuracy_test = accuracy_score(y_test, predicted_test)

    # # save predictions with random weights for comparison
    # initial_predictions = predicted_test

    # loss = cost(params, X_test, y_test, state_labels)

    # print(
    #     "Epoch: {:2d} | Cost: {:3f} | Train accuracy: {:3f} | Test Accuracy: {:3f}".format(
    #         0, loss, accuracy_train, accuracy_test
    #     )
    # )

    for it in range(epochs):
        for Xbatch, ybatch in iterate_minibatches(X_train, y_train, batch_size=batch_size):
            variables, cost_num = opt.step_and_cost(cost, params, Xbatch, ybatch, state_labels)
            params = variables[0]
            
        # predicted_train, fidel_train = test(params, X_train, y_train, state_labels)
        # accuracy_train = accuracy_score(y_train, predicted_train)
        # loss = cost(params, X_train, y_train, state_labels)

        # predicted_test, fidel_test = test(params, X_test, y_test, state_labels)
        # accuracy_test = accuracy_score(y_test, predicted_test)
        # res = [it + 1, loss, accuracy_train, accuracy_test]
        # print(
        #     "Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Test accuracy: {:3f}".format(
        #         *res
        #     )
        # )
        print(f'Epoch: {it + 1} | Cost: {cost_num}')

    predicted_test, fidel_test = test(params, X_test, y_test, state_labels)
    accuracy_test = accuracy_score(y_test, predicted_test)
           
    # Image plotting
    fig = plt.figure(figsize=(11.2, 5))
    fig.suptitle(f"Truth vs Predicted - parallel_uploading, L = {num_layers}, epochs = {epochs}, acc = {accuracy_test}")

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(list(zip(*X_test))[0], list(zip(*X_test))[1], 5, y_test)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(list(zip(*X_test))[0], list(zip(*X_test))[1], 5, predicted_test)

    file_name = f'{datetime.now().strftime("%d-%m-%Y %H-%M-%S")} - parallel_uploading, L = {num_layers}, epochs = {epochs}, acc = {accuracy_test}'
    plt.savefig(os.path.join(os.path.dirname(__file__), f'Plots\\{file_name}.png'))
    plt.show()
