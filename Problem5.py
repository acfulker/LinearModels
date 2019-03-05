import random

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_classification


def trainSGDSVM(X, y, T, eta=0.01):
    # learn a linear SVM by implementing the SGD algorithm
    #
    # Inputs:
    # X = N x d
    # y = N x 1
    # T = number of iterations
    # eta = learning rate
    # Output:
    # weight vector, w = d x 1

    # IMPLEMENT THIS METHOD
    w = np.zeros([X.shape[1], 1])

    for t in range(1, T):
        for i in range(1, X.shape[0]):
            val = y[i,:] * w.T
            check = np.matmul(val, X[i,:])
            if not check[0] < 1:
                continue
            error=y[i,:]*X[None,i,:]
            w=w+eta*error.T

    return w

if __name__ == "__main__":
    Xtrain, ytrain, Xtest, ytest = np.load(open('sample.pickle', 'rb'))
    # add intercept
    Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
    Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

    args = (Xtrain_i, ytrain)
    w = trainSGDSVM(Xtrain_i, ytrain, 100, 0.01)
    acc = evaluateLinearModel(w, Xtrain_i, ytrain)
    print('SVM Accuracy on train data - %.2f' % acc)
    acc = evaluateLinearModel(w, Xtest_i, ytest)
    print('SVM Accuracy on test data - %.2f' % acc)