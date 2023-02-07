# Loads and the simple 2D data to be used in QCNN and Hierarchical Classifier Training
import numpy as np

def data_load_and_process(dataset, n_train=2000, n_test=2000, classes=[0, 1], feature_reduction='resize256', binary=True):
    X_train, X_test = np.random.rand(n_train,2)*2-1,  np.random.rand(n_test,2)*2-1

    if dataset == 'circle':
        Y_train = ((np.sum(X_train**2, axis=1) > (3/4)**2).astype(int)*2-1).tolist()
        Y_test  = ((np.sum( X_test**2, axis=1) > (3/4)**2).astype(int)*2-1).tolist()
    
        return X_train, X_test, Y_train, Y_test
    
    if dataset == 'sinus1':
        Y_train = ((  X_train[:,1] > np.cos(3*X_train[:,0])  ).astype(int)*2-1).tolist()
        Y_test  = ((   X_test[:,1] > np.cos(3* X_test[:,0])  ).astype(int)*2-1).tolist()
    
        return X_train, X_test, Y_train, Y_test
    
    if dataset == 'sinus2':
        Y_train = ((  X_train[:,1] > -np.sin(3*X_train[:,0]) ).astype(int)*2-1).tolist()
        Y_test  = ((   X_test[:,1] > -np.sin(3* X_test[:,0]) ).astype(int)*2-1).tolist()
    
        return X_train, X_test, Y_train, Y_test
