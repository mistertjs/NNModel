# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:31:07 2016

@author: Administrator
"""

import numpy as np
rng = np.random
import NNFunctions as nnf

class NNLayer(object):

    # Activation Types
    A_Linear = 0
    A_Sigmoid = 1
    A_Tanh = 2
    A_Softmax = 3
    A_ReLU = 4
    
    # Layer Types
    T_Input = 0
    T_Hidden = 1
    T_Output = 2
    
    # activation (0=Linear, 1=Sigmoid, 2=Tanh, 3=Softmax)   
    # type (0=Input, 1=Hidden, 2=Output)
    def __init__(self, numNodes=1, activation=A_Linear, layerType=T_Hidden,
                 reluThreshold=0.0):
        self.numNodes = numNodes
        self.activation = activation
        self.layerType = layerType
        
        self.activationInput = None
        self.derivativeOutput = None
        self.activationOutput = None
        
        # backpropigation error for gradient calculation
        self.backpropError = None

        # rectifier threshold        
        self.reluThreshold = reluThreshold
        
    def getNodeCnt(self):
        return self.numNodes    
        
    def getActivationType(self):
        return self.activation
        
    def isInput(self):
        return (self.layerType == NNLayer.T_Input)

    def isHidden(self):
        return (self.layerType == NNLayer.T_Hidden)

    def isOutput(self):
        return (self.layerType == NNLayer.T_Output)
        
    def getActivationOutput(self):
        return self.activationOutput
        
    # Set output as X for 'Input' type layer. Others are calculated
    def setActivationOutput(self, X):
        self.activationInput = X        
        self.activationOutput = X
        
    def getDerivativeOutput(self):
        return self.derivativeOutput    
        
    def setBackpropError(self, Error):
        self.backpropError = Error
        
    def getBackpropError(self):
        return self.backpropError

    def getReLUThreshold(self):
        return self.reluThreshold
        
    '''
    Define the output activation functions
            
    # Define the logistic function
    def logistic(self, z):
        return 1 / (1 + np.exp(-z))
        
    # Define the softmax function
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    '''
    '''
    Cost Functions
    '''        
    def getCost(self, T):
        if (self.activation == NNLayer.A_Linear):
            return nnf.costLinear(self.activationOutput, T)
        # check for classificatoin
        elif (self.activation == NNLayer.A_Sigmoid):
            return nnf.costLogistic(self.activationOutput, T)
        elif (self.activation == NNLayer.A_Tanh):
            return nnf.costTanh(self.activationOutput, T)
        elif (self.activation == NNLayer.A_Softmax):
            return nnf.costSoftmax(self.activationOutput, T)
        elif (self.activation == NNLayer.A_ReLU):           
            '''
            This function should not be called, ReLU should not be an output
            '''
            return nnf.costReLU(self.activationOutput, T)
        
        return None

    '''
    Activation Outputs
    '''        
    # Functions to compute the hidden activations
    def calcActivation(self, X, W, b, retainOutput=True):
        Y = None
        # check for regression        
        if (self.activation == NNLayer.A_Linear):
            Y = nnf.linearActivation(X, W, b)
        # check for classification
        elif (self.activation == NNLayer.A_Sigmoid):
            Y = nnf.logisticActivation(X, W, b)
        elif (self.activation == NNLayer.A_Tanh):
            Y = nnf.tanhActivation(X, W, b)
        elif (self.activation == NNLayer.A_Softmax):
            Y = nnf.softmaxActivation(X, W, b)
        elif (self.activation == NNLayer.A_ReLU):
            Y = nnf.reluActivation(X, W, b, self.reluThreshold)
        
        # keep output in layer
        if (retainOutput):
            self.activationOutput = Y
            # keep input
            self.activationInput = X.dot(W) + b            
            
        return Y               

    # using the activation output and target output T, find the derivatives        
    def calcDerivatives(self, T):
        if (self.activation == NNLayer.A_Linear):
            self.derivativeOutput = nnf.linearDerivative(self.activationInput)
        # check for sigmoid y(1-y)
        elif (self.activation == NNLayer.A_Sigmoid):
            self.derivativeOutput = nnf.logisticDerivative(self.activationOutput) 
        # check fo tanh 1 - tanh(y)^2                                                
        elif (self.activation == NNLayer.A_Tanh):
            self.derivativeOutput = nnf.tanhDerivative(self.activationOutput)
        # check for softmax with 1 or multiple outputs
        elif (self.activation == NNLayer.A_Softmax):
            self.derivativeOutput = nnf.softmaxDerivative(self.activationInput,
                                                          T, self.isOutput())
        elif (self.activation == NNLayer.A_ReLU):
            self.derivativeOutput = nnf.reluDerivative(self.activationInput,
                                                       self.reluThreshold)
        
        return self.derivativeOutput
        
   