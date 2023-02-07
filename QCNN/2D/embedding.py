# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates.state_preparations import MottonenStatePreparation
import numpy as np
from Angular_hybrid import Angular_Hybrid_4, Angular_Hybrid_2
from global_var import *

#n_qubits has to be a 2^m number, where m is any positive integer
def data_embedding(X, embedding_type='Amplitude'):
    if embedding_type == 'Amplitude':
        AmplitudeEmbedding(X, wires=range(n_qbits), normalize=True)
    elif embedding_type == 'Angle':
        AngleEmbedding(X, wires=range(n_qbits), rotation='Y')
    elif embedding_type == 'Angle-compact':
        AngleEmbedding(X[:8], wires=range(n_qbits), rotation='X')
        AngleEmbedding(X[8:16], wires=range(n_qbits), rotation='Y')

    # Hybrid Direct Embedding (HDE)
    elif embedding_type == 'Amplitude-Hybrid4':

        for i in range(n_qbits//4):
            Xi = X[i * 2**4 : (i+1) * 2**4]
            normXi = np.linalg.norm(Xi)
            MottonenStatePreparation(Xi/normXi, wires=range(i*4, (i+1)*4))

    elif embedding_type == 'Amplitude-Hybrid2':

        for i in range(n_qbits//2):
            Xi = X[i * 2**2 : (i+1) * 2**2]
            normXi = np.linalg.norm(Xi)
            MottonenStatePreparation(Xi/normXi, wires=range(i*2, (i+1)*2))
            
            
    # Hybrid Angle Embedding (HAE)
    elif embedding_type == 'Angular-Hybrid4':
        N = 2**4 - 1  # 15 classical data in 4 qubits

        for i in range(n_qbits//4):
            Xi = X[i*N : (i+1)*N]
            Angular_Hybrid_4(Xi, wires=range(i*4, (i+1)*4))


    elif embedding_type == 'Angular-Hybrid2':
        N = 2**2 - 1   # 3 classical bits in 2 qubits
        
        for i in range(n_qbits//2):
            Xi = X[i*N : (i+1)*N]
            Angular_Hybrid_2(Xi, wires=range(i*2, (i+1)*2))





