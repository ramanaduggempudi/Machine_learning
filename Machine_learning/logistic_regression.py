import numpy as np

def sigmoid(x):
    # define the sigmoid function
    return 1/(1 + np.exp(-x))

def loss(y_pred, y, m):
    # define the loss function
    j = -np.sum(y*np.log(y_pred)) + (1-y)*(np.log(1-y))/m
    return j

def gradients():