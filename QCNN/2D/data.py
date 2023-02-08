# Loads and the simple 2D data to be used in QCNN and Hierarchical Classifier Training
import numpy as np
from global_var import *

def data_load_and_process(dataset, classes=[0, 1], feature_reduction='resize256', binary=True):
    X_train, X_test = np.random.rand(n_train,2)*2-1,  np.random.rand(n_test,2)*2-1

    if dataset == 'circle':
        Y_train = ((np.sum(X_train**2, axis=1) > (3/4)**2).astype(int)*2-1).tolist()
        Y_test  = ((np.sum( X_test**2, axis=1) > (3/4)**2).astype(int)*2-1).tolist()
    
    elif dataset == 'sinus1':
        Y_train = ((  X_train[:,1] > np.cos(3*X_train[:,0])  ).astype(int)*2-1).tolist()
        Y_test  = ((   X_test[:,1] > np.cos(3* X_test[:,0])  ).astype(int)*2-1).tolist()
    
    elif dataset == 'sinus2':
        Y_train = ((  X_train[:,1] > -np.sin(3*X_train[:,0]) ).astype(int)*2-1).tolist()
        Y_test  = ((   X_test[:,1] > -np.sin(3* X_test[:,0]) ).astype(int)*2-1).tolist()
    
    else:
        print("Incorrect dataset")
        return False
    
    
    if binary == False:
        Y_train = [1 if y == classes[0] else 0 for y in Y_train]
        Y_test = [1 if y == classes[0] else 0 for y in Y_test]
    elif binary == True:
        Y_train = [1 if y == classes[0] else -1 for y in Y_train]
        Y_test = [1 if y == classes[0] else -1 for y in Y_test]
        
    return X_train, X_test, Y_train, Y_test