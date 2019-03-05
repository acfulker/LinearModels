import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_classification
import math


def regressionObjVal(w, X, y):
    # compute squared error (scalar) with respect
    # to w (vector) for the given data X and y
    #
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # error = scalar value

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    n = X.shape[0]
    sum = 0
    for i in range(1, n):
        rhs = np.matmul(w.T, X[i])
        sum = sum + math.pow(y[i] - rhs, 2)
    error = 1 / 2 * sum
    print(error)
    return error


def regressionGradient(w, X, y):
    # compute gradient of squared error (scalar) with respect
    # to w (vector) for the given data X and y

    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # gradient = d length vector (not a d x 1 matrix)

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    if len(w.shape) == 1:
        w = w[:,np.newaxis]
    _x = np.matmul(X.T, X)
    lhs = np.matmul(_x, w)
    rhs = np.matmul(X.T, y)
    retval = lhs - rhs
    error_grad = retval.reshape(-1)
    print(error_grad.shape)
    return error_grad


def predictLinearModel(w, Xtest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # Output:
    # ypred = N x 1 vector of predictions

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    n = Xtest.shape[0]
    ypred = np.zeros([Xtest.shape[0], 1])
    for i in range(n):
        check = np.matmul(w.T, Xtest[i])
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


if __name__ == '__main__':
    Xtrain, ytrain, Xtest, ytest = np.load(open('sample.pickle', 'rb'))
    # add intercept
    Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
    Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

    args = (Xtrain_i, ytrain)
    opts = {'maxiter': 50}  # Preferred value.
    w_init = np.zeros((Xtrain_i.shape[1], 1))
    soln = minimize(regressionObjVal, w_init, jac=regressionGradient, args=args, method='CG', options=opts)
    w = np.transpose(np.array(soln.x))
    w = w[:, np.newaxis]
    acc = evaluateLinearModel(w, Xtrain_i, ytrain)
    print('Perceptron Accuracy on train data - %.2f' % acc)
    acc = evaluateLinearModel(w, Xtest_i, ytest)
    print('Perceptron Accuracy on test data - %.2f' % acc)
