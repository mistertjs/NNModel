# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:53:59 2016

@author: tschlosser
"""

import numpy as np
rng = np.random
from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from copy import deepcopy

class NNGradientTest(object):
    """
    Initialize the gradient model or pass a customized model to the checker    
    """
    
    def __init__(self, nnModel=None, nnTestData=None):
        """
        Define the model structure
        """
        self.nnModel = nnModel
        if (nnModel is None):
            l1 = NNLayer(3,activation=NNLayer.A_Linear, layerType=NNLayer.T_Input)
            l2 = NNLayer(2,activation=NNLayer.A_Linear, layerType=NNLayer.T_Hidden)
            l3 = NNLayer(1,activation=NNLayer.A_Linear, layerType=NNLayer.T_Output)
            c1 = NNConnection(l1,l2)
            c2 = NNConnection(l2,l3)
            #c1 = NNConnection(l1,l3)
            self.nnModel = NNModel()
            self.nnModel.addConnection(c1)
            self.nnModel.addConnection(c2)
        
            """
            Create the test data using the number of nodes in the InputLayer for the col
            dimension of the X input
            """
            # test data
            self.m = 100
            self.n = nnModel.getInputLayer().getNodeCnt()
            self.x = rng.rand(self.m,self.n)
            # the larger
            self.w = np.linspace(0,3,self.n) + 1.0
            self.b = 3.5
            #err = rng.randn(m)
            z = np.dot(self.x,np.transpose(self.w))
            self.y = np.asmatrix(z + self.b).reshape(self.m,1)
        else:
            if (nnTestData is None):
                raise ValueError("TestData must be supplied when not using a default nnModel value")
            # transfer test data to gradient tester
            self.m = nnTestData.m            
            self.n = nnTestData.n
            self.x = nnTestData.x
            self.w = nnTestData.w
            self.b = nnTestData.b
            self.y = nnTestData.y

    """
    Run the gradient check test with default or customized hyper-parameters
    WARNING: Regularization strength will impact the values, so it needs to be
    very low or set to zero
    """
    def runTest(self, alpha=0.001, eps=0.001, regStrength=0.0, printResults=False):
        """
        Initialize the model hyper-parameters and run the model to get the resultant
        gradients for each model parameter
        """
        # get initial copy of model params
        params = self.nnModel.getModelParams()
        c_params = deepcopy(params)
        
        # Run the model through one iteration forward and backwards
        self.nnModel.forwardPropigation(self.x,self.y)
        self.nnModel.backPropigation(self.y,alpha,regStrength)
        grads = self.nnModel.getModelGradients()
        
        """
        Test the backpropigation using the epsilon value
        """
        self.passed = True
        eps = 0.001
        # iterate through each model parameter and compare its gradient with cost delta            
        for l in range(len(c_params)):
            # get param list for this connection with 2 objects W,b
            for obj in range(len(c_params[l])):
                p = c_params[l][obj]
                for row in range(p.shape[0]):
                    for col in range(p.shape[1]):
                        # update the parameter
                        params = deepcopy(c_params)
                        params[l][obj][row,col] += eps
                        # run the model and get cost
                        self.nnModel.setModelParams(params)
                        self.nnModel.forwardPropigation(self.x,self.y)
                        self.nnModel.backPropigation(self.y,alpha,regStrength)
                        costPlus = self.nnModel.getCost(self.y)
                        
                       # update the parameter
                        params = deepcopy(c_params)
                        params[l][obj][row,col] -= eps
                        # run the model and get cost
                        self.nnModel.setModelParams(params)
                        self.nnModel.forwardPropigation(self.x,self.y)
                        self.nnModel.backPropigation(self.y,alpha,regStrength)
                        costMinus = self.nnModel.getCost(self.y)
                        
                        # get gradient
                        grad_num = grads[l][obj][row,col]
                        grad_res = (costPlus - costMinus)/(2*eps)
                        #
                        
                        # Raise error if the numerical grade is not close to the backprop gradient
                        if not np.isclose(grad_num, grad_res, atol=1e-05):
                            print "Layer:%d Weights(%d,%d): %3.6f = %3.6f" % (l,row,col,grad_num, grad_res)
                            #print 'Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!' % (float(grad_num), float(grad_res))
                            self.passed = False
                        elif(printResults):
                            if (obj == 0):
                                print "Layer:%d Weights(%d,%d): %3.6f = %3.6f" % (l,row,col,grad_num, grad_res)
                            else:
                                print "Layer:%d Bias(%d,%d): %3.6f = %3.6f" % (l,row,col,grad_num, grad_res)
                
                
        # Return the result                
        return self.passed                
            