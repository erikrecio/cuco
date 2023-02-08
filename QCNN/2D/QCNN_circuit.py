import pennylane as qml
from unitary import *
import embedding
import numpy as np
from global_var import *


def QCNN_structure(U, V, params):
    layers = int(np.log2(n_qbits))
    U_params = dic_U_params[U]
    V_params = dic_U_params[V]
    U = eval(U)
    V = eval(V)
    
    for j in range(layers):
        
        Uparams = params[j*U_params : (j+1)*U_params]
        Vparams = params[layers*U_params + j*V_params : layers*U_params + (j+1)*V_params]
        
        for i in range(0, n_qbits, 2**(j+1)):
            U(Uparams, wires=[i, i + 2**j])
        if j != layers-1:
            U(Uparams, wires=[n_qbits - 2**j, 0])
        for i in range(2**j, n_qbits - 2**j, 2**(j+1)):
            U(Uparams, wires=[i, i + 2**j])

        if j != layers-1:
            for i in range(0, n_qbits, 2**(j+1)):
                V(Vparams, wires=[i+2**j, i])
        else:
            V(Vparams, wires=[0, n_qbits//2])
            
            

def QCNN_structure_without_pooling(U, params):
    layers = int(np.log2(n_qbits))
    U_params = dic_U_params[U]
    U = eval(U)
    
    for j in range(layers):
        
        Uparams = params[j*U_params : (j+1)*U_params]
        
        for i in range(0, n_qbits, 2**(j+1)):
            U(Uparams, wires=[i, i + 2**j])
        if j != layers-1:
            U(Uparams, wires=[0, n_qbits - 2**j])
        for i in range(2**j, n_qbits - 2**j, 2**(j+1)):
            U(Uparams, wires=[i, i + 2**j])
    
    


def QCNN_1D_circuit(U, params):
    
    layers = int(np.log2(n_qbits))
    U_params = dic_U_params[U]
    U = eval(U)
    
    for j in range(layers):
        
        Uparams = params[j*U_params : (j+1)*U_params]
        
        for i in range(0, n_qbits, 2**(j+1)):
            U(Uparams, wires=[i, i + 2**j])
        for i in range(2**j, n_qbits - 2**j, 2**(j+1)):
            U(Uparams, wires=[i, i + 2**j])
            
            



dev = qml.device(machine_type, wires = n_qbits)
@qml.qnode(dev)
def QCNN(X, params, U, V, struct, embedding_type='Amplitude', cost_fn='cross_entropy'):

    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)

    # Quantum Convolutional Neural Network
    if struct == "default":
        QCNN_structure(U, V, params)
    elif struct == 'no_pooling':
        QCNN_structure_without_pooling(U, params)
    elif struct == 'no_pooling_1D':
        QCNN_1D_circuit(U, params)
    else:
        print("Invalid Unitary Ansatze")
        return False

    if cost_fn == 'mse':
        result = qml.expval(qml.PauliZ(n_qbits//2))
    elif cost_fn == 'cross_entropy':
        result = qml.probs(wires=n_qbits//2)
    return result