import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_classification
import math

def logisticObjVal(w, X, y):
    # compute log-loss error (scalar) with respect
    # to w (vector) for the given data X and y
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # error = scalar
    #print(w.shape[0])
    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    w = w[:, np.newaxis]
    n = X.shape[0]
    error = 0
    sum = 0

    for i in range(n):
        yi=y[i]
        yi=yi[:,np.newaxis]
        wt=w.T
        xi=X[i,:]
        xi=xi[:,np.newaxis]
        val = np.matmul(yi, wt)
        e_val = np.matmul(val, xi)
        log_val = np.log(1 + math.exp(-e_val))
        sum = sum + log_val
    error = 1/n * sum
    return error

def logisticGradient(w, X, y):

    # compute the gradient of the log-loss error (vector) with respect
    # to w (vector) for the given data X and y
    #
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # error = d length gradient vector (not a d x 1 matrix)
    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    n = X.shape[0]
    w = w[:,np.newaxis]
    wt=w.T
    sum = 0
    for i in range(n):
        xi=X[i,:]
        xi=xi[:,np.newaxis]
        yi=y[i]
        yi=yi[:,np.newaxis]
        val = np.matmul(yi, wt)
        e_val = np.matmul(val, xi)
        down = 1 + np.exp(e_val)
        expression=yi/down
        res=expression*xi
        res=res.reshape(-1)
        sum = sum + res
        
        
    gradient = -1/n * sum
    return gradient


def logisticHessian(w, X, y):
    # compute the Hessian of the log-loss error (matrix) with respect
    # to w (vector) for the given data X and y
    #
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # Hessian = d x d matrix
    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    w = w[:, np.newaxis]
    wt=w.T
    n = X.shape[0]
    sum = 0
    for i in range(n):
        yi=y[i]
        yi=yi[:,np.newaxis]
        xi=X[i,:]
        xi = xi[:, np.newaxis]
        val = np.matmul(yi, wt)
        e_val = np.matmul(val, xi)
        down = 1 + math.exp(e_val)
        down_sq = math.pow(down, 2)
        up = math.exp(e_val)
        expression=up/down_sq
        out= np.matmul(xi,xi.T)
        full=expression*out
        sum = sum + full
    hessian = 1/n * sum
    return hessian

if __name__ == '__main__':
    Xtrain, ytrain, Xtest, ytest = np.load(open('sample.pickle', 'rb'))
    # add intercept
    Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
    Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

    args = (Xtrain_i, ytrain)
    opts = {'maxiter': 50}  # Preferred value.
    w_init = np.zeros((Xtrain_i.shape[1], 1))
    soln = minimize(logisticObjVal, w_init, jac=logisticGradient, hess=logisticHessian, args=args, method='Newton-CG',
                    options=opts)
    w = np.transpose(np.array(soln.x))
    w = np.reshape(w, [len(w), 1])
    acc = evaluateLinearModel(w, Xtrain_i, ytrain)
    print('Logistic Regression Accuracy on train data - %.2f' % acc)
    acc = evaluateLinearModel(w, Xtest_i, ytest)
    print('Logistic Regression Accuracy on test data - %.2f' % acc)