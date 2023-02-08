# This generates the results of the bechmarking code

import Benchmarking


"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]
Encodings: ['resize256', 'pca8', 'autoencoder8', 'pca16-compact', 'autoencoder16-compact', 'pca32-1', 'autoencoder32-1',
            'pca16-1', 'autoencoder16-1', 'pca30-1', 'autoencoder30-1', 'pca12-1', 'autoencoder12-1']
dataset: 'mnist' or 'fashion_mnist'
circuit: 'QCNN' or 'Hierarchical'
cost_fn: 'mse' or 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""


    # 'resize256'               - 'Amplitude'
    # 'pca8'                    - 'Angle'
    # 'autoencoder8'            - 'Angle'
    # 'pca16-compact'           - 'Angle-compact'
    # 'autoencoder16-compact'   - 'Angle-compact'
    
    # # Amplitude Hybrid Embedding
    # # 4 qubit block
    # 'pca32'                   - 'Amplitude-Hybrid4'
    # 'autoencoder32'           - 'Amplitude-Hybrid4'

    # # 2 qubit block
    # 'pca16'                   - 'Amplitude-Hybrid2'
    # 'autoencoder16'           - 'Amplitude-Hybrid2'

    # # Angular Hybrid Embedding
    # # 4 qubit block
    # 'pca30'                   - 'Angular-Hybrid4'
    # 'autoencoder30'           - 'Angular-Hybrid4'

    # # 2 qubit block
    # 'pca12'                   - 'Angular-Hybrid2'
    # 'autoencoder12'           - 'Angular-Hybrid2'

# Paper
Unitaries = ['U_SU4']           #['U_SU4', 'U_SU4', 'U_SU4_no_pooling', 'U_9_1D']
Vtaries = ["Pooling_ansatz1"]   #["Pooling_ansatz1", "Pooling_ansatz1", "Pooling_ansatz1", "Pooling_ansatz1"]
Structs = ["default"]           #["default", "default", "no_pooling", "no_pooling_1D"]
Encodings = ['resize256']
dataset = 'mnist'              #'mnist', 'circle'
classes = [0,1]
binary = False
cost_fn = 'cross_entropy'

#2D
Unitaries = ['U_SU4']           #['U_SU4', 'U_SU4', 'U_SU4_no_pooling', 'U_9_1D']
Vtaries = ["Pooling_ansatz1"]   #["Pooling_ansatz1", "Pooling_ansatz1", "Pooling_ansatz1", "Pooling_ansatz1"]
Structs = ["default"]           #["default", "default", "no_pooling", "no_pooling_1D"]
Encodings = ['pca8']
dataset = 'circle'              #'mnist', 'circle'
classes = [0,1]
binary = False
cost_fn = 'cross_entropy'


Benchmarking.Benchmarking(dataset, classes, Unitaries, Vtaries, Structs, Encodings, circuit='QCNN', cost_fn=cost_fn, binary=binary)
#Benchmarking.Benchmarking(dataset, classes, Unitaries, Encodings, circuit='Hierarchical', cost_fn=cost_fn, binary=binary)

