import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_classification
import math

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
    for i in range(1, N):
        lhs = ytest[i]
        rhs = np.matmul(w.T, Xtest[i])
        jw = jw + math.pow(lhs - rhs, 2)
    rmse = math.sqrt(1/Xtest.shape[0] * jw)
    return rmse

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

    n = X.shape[0]
    sum = 0
    for i in range(1, n):
        rhs = np.matmul(w.T, X[i])
        sum = sum + math.pow(y[i] - rhs, 2)
    error = 1/2 * sum

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
    if len(w.shape) == 1:
        w = w[:,np.newaxis]
    _x = np.matmul(X.T, X)
    lhs = np.matmul(_x, w)
    rhs = np.matmul(X.T, y)
    retval = lhs - rhs
    error_grad = retval.reshape(-1)

    return error_grad

if __name__ == '__main__':
    Xtrain, ytrain, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')
    # add intercept
    Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
    Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)
    args = (Xtrain_i, ytrain)
    opts = {'maxiter': 50}  # Preferred value.
    w_init = np.zeros((Xtrain_i.shape[1], 1))
    soln = minimize(regressionObjVal, w_init, jac=regressionGradient, args=args, method='CG', options=opts)
    w = np.transpose(np.array(soln.x))
    w = w[:, np.newaxis]
    rmse = testOLERegression(w, Xtrain_i, ytrain)
    print('Gradient Descent Linear Regression RMSE on train data - %.2f' % rmse)
    rmse = testOLERegression(w, Xtest_i, ytest)
    print('Gradient Descent Linear Regression RMSE on test data - %.2f' % rmse)