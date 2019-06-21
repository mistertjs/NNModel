# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:32:21 2016

@author: Administrator
"""

import numpy as np
rng = np.random
from copy import deepcopy

class NNConnection(object):

    # provide list of layers to connect            
    def __init__(self, inputLayer, outputLayer):
        self.Weights = None
        self.Bias = None
     
        self.inputLayer = inputLayer
        self.outputLayer = outputLayer

        # create mulitdimensional lists for [input],[output] layers
        numInputs = inputLayer.getNodeCnt()
        numOutputs = outputLayer.getNodeCnt()
        self.Weights = 0.01 * rng.randn(numInputs, numOutputs)
        # Test weights
        # self.Weights = np.matrix([(10.0,20.0,30.)]).reshape(numInputs, numOutputs)
        self.Bias = np.zeros((1,numOutputs))
        
        self.g_W = None
        self.g_b = None
        self.g_WRatio = []
        self.g_WVar = []
        
    def getShape(self):
        return np.shape(self.Weights)
        
    def getInputLayer(self):
        return self.inputLayer

    def getOutputLayer(self):                        
        return self.outputLayer
        
    def setWeights(self, W):
        self.Weights = np.copy(W)
        
    def getWeights(self):
        return self.Weights

    def setBias(self, b):
        self.Bias = np.copy(b)
        
    def getBias(self):
        return self.Bias
        
    def hasInput(self):
        return self.inputLayer.isInput()
        
    def getInputCnt(self):
        return len(self.inputLayer)
        
    def hasOutput(self):
        return self.outputLayer.isOutput()
        
    def getOutputCnt(self):
        return len(self.outputLayer)     
        
    def getGradients(self):
        return [self.g_W, self.g_b]    

    def setParams(self, W, b, clearGradients=True):
        self.Weights = deepcopy(W)
        self.Bias = deepcopy(b)
        if (clearGradients):
            self.g_W = None
            self.g_b = None
            del self.g_WRatio[:]

    def getParams(self, copy=True):
        if (copy):
            return deepcopy([self.Weights, self.Bias])
        else:
            return [self.Weights, self.Bias]