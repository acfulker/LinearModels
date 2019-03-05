import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_classification
import math

def learnOLERegression(X, y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    trans_X = X.T
    XTX = np.matmul(trans_X, X)
    inverse = np.linalg.inv(XTX)
    new_X = np.matmul(inverse, trans_X)
    w = np.matmul(new_X, y)

    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # rmse = scalar value

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    N = Xtest.shape[0]
    jw = 0
    wt=w.T
    for i in range(0, N-1):
        lhs = ytest[i]
        xi=Xtest[i]
        xi=xi[:,np.newaxis]
        rhs =np.matmul(wt,xi)
        jw = jw + math.pow(lhs - rhs, 2)
    rmse = math.sqrt(1/Xtest.shape[0] * jw)
    return rmse

if __name__ == '__main__':
    Xtrain, ytrain, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')
    # add intercept
    x1 = np.ones((len(Xtrain), 1))
    x2 = np.ones((len(Xtest), 1))

    Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
    Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

    w = learnOLERegression(Xtrain, ytrain)
    w_i = learnOLERegression(Xtrain_i, ytrain)

    rmse = testOLERegression(w, Xtrain, ytrain)
    rmse_i = testOLERegression(w_i, Xtrain_i, ytrain)
    print('RMSE without intercept on train data - %.2f' % rmse)
    print('RMSE with intercept on train data - %.2f' % rmse_i)

    rmse = testOLERegression(w, Xtest, ytest)
    rmse_i = testOLERegression(w_i, Xtest_i, ytest)
    print('RMSE without intercept on test data - %.2f' % rmse)
    print('RMSE with intercept on test data - %.2f' % rmse_i)