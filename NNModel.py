# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:32:58 2016

@author: Administrator
"""
import numpy as np
from NNLayer import NNLayer
from pprint import pprint
from copy import deepcopy
rng = np.random

class NNModel(object):
    def __init__(self):
        self.Connections = []
        self.InputConnection = None
        self.OutputConnection = None
        
    def getConnectionCnt(self):
        return len(self.Connections)        
        
    def getOutputType(self):
        return self.OutputConnection.getOutputLayer().getActivationType()
                
    def getModelParams(self):
        params = []
        for con in self.Connections:
            params.append(con.getParams())
        return params
        
    def getModelGradients(self):
        grads = []
        for con in self.Connections:
            grads.append(con.getGradients())
        return grads
        
    def setModelParams(self, paramList):
        for i in range(self.getConnectionCnt()):
            param = paramList[i]
            con = self.Connections[i]
            W = deepcopy(param[0])
            b = deepcopy(param[1])
            # set the connection parameters and clear the gradient list
            con.setParams(W, b)
       
    def addConnection(self, connection):
        # capture the input connection 
        if (connection.hasInput()):
            self.InputConnection = connection
        if (connection.hasOutput()):
            self.OutputConnection = connection
            
        # add to connection list            
        self.Connections.append(connection)   
        
    def getInputLayer(self):
        con = self.Connections[0]
        return con.getInputLayer()
        
    def getOutputLayer(self):
        return self.OutputConnection.getOutputLayer()
        
    def forwardPropigation(self, X, T):
        Y = None
        # set the input layer with X
        for con in self.Connections:
            # get input and output layers
            inputLayer = con.getInputLayer()
            outputLayer = con.getOutputLayer()

            # seed the input layer with input X            
            if (con.hasInput()):
                inputLayer.setActivationOutput(X)
            
            # calculate the output layer using the input layer output
            W = con.getWeights()
            b = con.getBias()
            # check if this layer is the input layer...if so, retain X,
            # otherwise pull from the connected input layer
            if (inputLayer.isInput() == False):
                X = inputLayer.getActivationOutput()            
            
            # calculate the output
            #print np.shape(W)
            Y = outputLayer.calcActivation(X, W, b)
            
            # calculate the output derivatives for backpropigation
            #outputLayer.calcDerivatives(T)

        # return output Y
        return Y

    def nnOutput(self, X):
        """ Return the output of the model without storing anything """  
       # set the input layer with X
        Y = None
        for con in self.Connections:
            # calculate the output layer using the input layer output
            W = con.getWeights()
            b = con.getBias()
            # check if this layer is the input layer...if so, retain X,
            # otherwise pull from the connected input layer
            if (Y is None):
                Y = np.copy(X)

            # calculate the output
            #print np.shape(W)
            Y = con.getOutputLayer().calcActivation(Y, W, b, False)

        # return output Y
        return Y

        
    def getCost(self, T, regStrength=None):
        # return the output nodes cost
        output = self.OutputConnection.getOutputLayer()
        
        # apply regularization
        regLoss = 0
        if (regStrength <> None):
            for con in self.Connections:
                W = con.getWeights()
                regLoss += 0.5 * regStrength * np.sum(W*W)
                
        dataLoss = output.getCost(T)   
        #print "Shape loss:%s T: %s" % (dataLoss.shape, T.shape)
        cost = dataLoss + regLoss
        
        return cost

    
    def getCostDerivativeComponent(self, T):
        """ Returns only the dE/dy portion for the delta"""
        # return the output nodes cost
        output = self.OutputConnection.getOutputLayer()
        outputActivationType = output.getActivationType()

        # get output layer output        
        Y = output.getActivationOutput()
        
        # check for regression        
        costDerivative = None
        if (outputActivationType == NNLayer.A_Linear):
            costDerivative = (Y - T)                    
        # return logistic component (t - y)/[y(1-y)]
        elif (outputActivationType == NNLayer.A_Sigmoid):
            costDerivative = np.divide((Y - T),
                                       np.multiply(Y, (1 - Y)))
        # check for classificatoin
        elif (output.getActivationType() == NNLayer.A_Tanh):
            costDerivative = np.divide((Y - T), (1 - np.square(Y)))
            
        # check for softmax            
        elif (output.getActivationType() == NNLayer.A_Softmax):
            #costDerivative = (Y - T) 
            #costDerivative = -np.multiply(T, np.log(Y)).sum()
            costDerivative = -np.divide(T, Y)
                                       
        return costDerivative
        
    def getOutputBackpropError(self, T):
        # get output of last layer
        outputLayer = self.OutputConnection.getOutputLayer()
        Y = outputLayer.getActivationOutput()

        # return fully calculated backprop error (dE/dz)        
        outputActivationType = outputLayer.getActivationType()
        costDerivative = None
        if (outputActivationType == NNLayer.A_Linear):
            costDerivative = (Y - T)                    
        # return logistic component (t - y)/[y(1-y)]
        elif (outputActivationType == NNLayer.A_Sigmoid):
            costDerivative = np.divide((Y - T),
                                       np.multiply(Y, (1 - Y)))
        # check for classificatoin
        elif (outputLayer.getActivationType() == NNLayer.A_Tanh):
            costDerivative = np.divide((Y - T), (1 - np.square(Y)))
            
        # check for softmax    
        outputBackpropError = None            
        if (outputActivationType == NNLayer.A_Softmax):
            #outputBackpropError = np.multiply(Y,T-1.) + np.multiply(Y,T) - T
            outputBackpropError = (Y - T) 
            #outputBackpropError = Y - 1.0
        else:
            AD = outputLayer.calcDerivatives(T)
            # multiply and sum
            outputBackpropError = np.multiply(AD, costDerivative)                
            
        return outputBackpropError            
        
    def getOutputBackpropError2(self, T):
        # get output of last layer
        outputLayer = self.OutputConnection.getOutputLayer()
        Y = outputLayer.getActivationOutput()

        # return fully calculated backprop error (dE/dz)        
        outputActivationType = outputLayer.getActivationType()
        # by default, all the compositional output layer error is
        # simply Y-T, except for ReLU which needs to be masked
        outputBackpropError = (Y - T)                    
        '''
        if (outputActivationType == NNLayer.A_ReLU):
           multMask = (Y > 0.).astype(dtype='float64') * 1.
           YMask = np.multiply(multMask, Y)
           outputBackpropError = (YMask - T)     
        '''    
        return outputBackpropError
        
    # pass the target output        
    def backPropigation(self, T, alpha, regStrength):
        
        '''
        # derive the first error set using the outtermost layer
        outputLayer = self.OutputConnection.getOutputLayer()
        if (outputLayer.getActivationType == NNLayer.A_Softmax):
            D = outputLayer.getActivationOutput() - T
        else:            
            AD = outputLayer.calcDerivatives(T)
            # get the output layer cost derivative component
            D = self.getCostDerivativeComponent(T)       
            # multiply and sum
            D = np.multiply(AD, D)
        '''            
        outputLayer = self.OutputConnection.getOutputLayer()
        # get output backprop error completely formed as opposed to a 
        # compositional calculation
        D = self.getOutputBackpropError2(T)            
        # add regularization derivative and apply to error
        W = self.OutputConnection.getWeights()
        #DW = np.zeros(D.shape)
        #if (regStrength <> 0):
        #    DW = regStrength * np.sum(W) #np.sum(np.dot(W.T,W))
        
        # set backprop error for the inputLayer
        #outputLayer.setBackpropError(D + DW)
        outputLayer.setBackpropError(D)
        #print "Layer out DShape: %s, Delta: %s" % (np.shape(D),D)
        
        # get Deltas into each node in reverse order
        maxConnections = len(self.Connections)
        #if (maxConnections > 1):
        for c in xrange(maxConnections-1,-1,-1):
            con = self.Connections[c]
            # get layers for this connectino
            outputLayer = con.getOutputLayer()
            inputLayer = con.getInputLayer()
            #print "Output Layer: %s" % outputLayer.getNodeCnt()
            
            '''
            Update error for Input layer
            '''
            # generate Delta from the output layer
            Dout = outputLayer.getBackpropError()
            # get connection weights
            W = con.getWeights()
            # get output layer activation derivatives            
            AD = inputLayer.calcDerivatives(T)
            #AD = inputLayer.getDerivativeOutput()
            # calculate input layer error as (m,k) matrix
            Din = np.dot(Dout, W.T)
            Din = np.multiply(Din, AD)
            # add regularization derivative
            #DW = 0.5 * regStrength * np.sum(np.dot(W.T,W))
            # add regularization derivative and apply to error
            W = self.OutputConnection.getWeights()
            #DW = np.zeros(Din.shape)
            #if (regStrength <> 0):
            #    DW = regStrength * np.sum(W) #np.sum(np.dot(W.T,W))
            
            # set backprop error for the inputLayer
            # print "Layer %s DShape: %s, Delta: %s" % (c, np.shape(D),D)
            con.getInputLayer().setBackpropError(Din)
            
            '''
            Update Weights for this connection with Output layer error
            '''
            # get input activation output
            A = inputLayer.getActivationOutput()
            # get gradients            
            #print "Shapes: %s %s" % (np.shape(A), np.shape(D))
            m = np.shape(T)[0]
            g_W = np.dot(A.T, Dout)/m + (regStrength * con.getWeights())
            g_b = np.sum(Dout, axis=0)/m

            # track the gradients
            con.g_W = g_W
            con.g_b = g_b
            con.g_WVar.append(g_W)
            #con.g_WRatio.append(np.sum(np.dot(g_W.T,g_W)))
            con.g_WRatio.append(np.sum(np.multiply(g_W,g_W)))
            

            # update weights
            W = con.getWeights()
            W -= alpha * g_W
            b = con.getBias()
            b -= alpha * g_b
            
            # set back the weights
            con.setWeights(W)
            con.setBias(b)
            
    def getGWVar(self):
        return self.OutputConnection.g_WVar 
            
    def showWeights(self, layer=None):
        if (layer is not None):
            W = self.Connections[layer].getWeights()
            b = self.Connections[layer].getBias()
            pprint("Layer: %s, Weights: %s, Bias: %s" % (layer,W,b))
        else:
            for layer in range(len(self.Connections)):
                W = self.Connections[layer].getWeights()
                b = self.Connections[layer].getBias()
                pprint("Layer: %s, Weights: (%s,%s), Bias: %s" % (layer,np.shape(W),np.around(W,2),b))
        

