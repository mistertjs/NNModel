# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 13:27:03 2016

@author: Administrator
"""
import numpy as np
rng = np.random

'''
Helper Functions
'''
# Define the logistic function
def logistic(z):
    return 1 / (1 + np.exp(-z))
    
# Define the softmax function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

'''
Cost Functions
'''
def costLinear(Y, T):
    return 0.5 * np.multiply(Y - T, Y - T).mean(axis=0)

# cost for sigmoid and tanh functions        
def costLogistic(Y, T):
    return -(np.multiply(T, np.log(Y)) +
             np.multiply(1 - T, np.log(1 - Y))).mean(axis=0)

def costTanh(Y, T):
    return -(np.multiply((T+1)/2, np.log((Y+1)/2)) +
             np.multiply(1 - (T+1)/2, np.log(1 - (Y+1)/2))).mean(axis=0)

# cross-entropy cost for Softmax
def costSoftmax(Y, T):
    # replace sum(axis=0) with sum()
    #return -np.sum(np.multiply(T, np.log(Y)),axis=0).mean()
    m = float(T.shape[0])
    return -np.multiply(T, np.log(Y)).sum()/m
    
def costReLU(Y, T):
    '''
    It's not a good idea to use ReLU activation for an output
    '''
    # get linear cost
    cost = 0.5 * np.multiply(Y - T, Y - T)
    # max cost values that have valid values
    costMask = (Y > 0.).astype(dtype='float64') * 1.
    return np.multiply(cost, costMask).mean()
        
             
'''
Activation Functions
'''             
def linearActivation(X, W, b):
    return X.dot(W) + b

def logisticActivation(X, W, b):
    return logistic(X.dot(W) + b)

def tanhActivation(X, W, b):
    return np.tanh(X.dot(W) + b)

def softmaxActivation(X, W, b):
    return softmax(X.dot(W) + b)

def reluActivation(X, W, b, threshold):
    return np.maximum(threshold, X.dot(W) + b).astype(dtype='float64')
    
'''
Activation Derivatives
Inputs to the functions have already undergone the activation function,
where the derivative seems to incorporate a component of the activation
e.g. Logistic s; dA/dZ = s(1-s)
     Tanh tanh; dA/dZ = (1 - tanh**2)
'''    

def linearDerivative(Y):
    return np.ones(Y.shape)
    
def logisticDerivative(Y):
    return np.multiply(Y, (1.0 - Y))
    
# Passes in the activation output which is already tanh(X)    
def tanhDerivative(Y):
    return 1.0 - np.square(Y)

def softmaxDerivative(Y, T, bIsOutput=True):
    rows = np.shape(Y)[0]        
    cols = np.shape(Y)[1]
    smOut = None
    if (cols == 1):
        # p - 1(y == k)
        # NOTE: This can only be an output node and layer
        smOut = Y[range(rows),T]
    # multiple columns
    else:
        # TBD...this is correct, only for an output type
        if (bIsOutput):
            #smOut = np.multiply(Y, T) - 1.0
            smOut = Y - T
        else:
            print "ERROR IN calcDerivatives...should be an 'output' type"
        
    return smOut

def reluDerivative(Y, threshold):
    '''
    Derivative of a ReLU is 1 if Y > threshold, otherwise 0
    Min value will be 1 * threshold
    '''
    #result = np.abs(np.divide(np.max((np.ones(Y.shape,dtype='float64') * float(threshold),Y),axis=0),Y))
    #result = np.divide(np.max((np.ones(Y.shape,dtype='float64') * threshold, Y),axis=0),Y)
    # replace nan with 0 since a divide by 0 occured    
    #return np.nan_to_num(result)
    multiplier = (Y > threshold).astype(dtype='float64') * 1.
    #return np.multiply(multiplier, Y)
    return multiplier
    
    
'''
Data splitting for mini batch and SGD
'''      
def splitData(X, Y, testRatio):
 # create training and test set
    m = len(Y)
    k = int(np.around(m * testRatio))
    # create a random array for selection
    rndIdx = np.random.choice(m, k)
    # test set
    yTest = Y[rndIdx]
    xTest = X[rndIdx]
    # train set
    yTrain = np.delete(Y, rndIdx, axis=0)
    xTrain = np.delete(X, rndIdx, axis=0)
    
    return [xTrain, yTrain, xTest, yTest]
    
def getClassifierError(Y, T):
    # output error
    tP = np.sum(T[np.where(Y==1)[0]])
    fN = np.sum(T[np.where(Y==0)[0]])
    #aP = np.sum(T)
    #aN = len(T) - aP
    eP = np.sum(Y)
    eN = len(Y) - eP
    #fP = eP - tP
    tN = eN - fN
    accuracy = (tN + tP)/len(Y)
    error = 1.0 - accuracy    
    return error
    
def getRegressionError(Y, T):
    np.dot((Y - T).T, (Y - T))