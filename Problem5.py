import random

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_classification


def predictLinearModel(w, Xtest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # Output:
    # ypred = N x 1 vector of predictions

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    n = Xtest.shape[0]
    ypred = np.zeros([Xtest.shape[0], 1])
    wt=w.T
    for i in range(n):
        xi=Xtest[i,:]
        xi=xi[:,np.newaxis]
        check = np.matmul(wt, xi)
        if check < 0:
            ypred[i] = -1
        elif check >= 0:
            ypred[i] = 1

    return ypred


def evaluateLinearModel(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # acc = scalar values

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    acc = 0
    n = Xtest.shape[0]
    ypred=predictLinearModel(w, Xtest)
    for i in range(n):
        if(ypred[i]==ytest[i]):
            acc+=1
    ret = acc/n        
    return ret

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
    sample=[]
    for i in range(X.shape[0]):
        if (np.random.random_sample()>.5):
            sample.append(i)
            
    for t in range(T):
        for i in sample:
            yi=y[i,:]
            yi=yi[:,np.newaxis]
            wt=w.T
            xi=X[i,:]
            xi=xi[:,np.newaxis]
            val = np.matmul(yi,wt)
            check = np.matmul(val, xi)
            check=check.reshape(-1)
            if not check[0] < 1:
                continue
            error=yi*xi
            w=w+eta*error

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