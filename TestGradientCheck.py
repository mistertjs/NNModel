# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 06:30:48 2016

@author: Administrator
"""

#import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
#sys.path.append('C:/Users/Administrator/Google Drive/Python\NNModel')

import numpy as np
import matplotlib.pyplot as plt
rng = np.random
from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from copy import deepcopy

"""
Define the model structure
"""
l1 = NNLayer(3,activation=NNLayer.A_Linear, layerType=NNLayer.T_Input)
l2 = NNLayer(2,activation=NNLayer.A_Linear, layerType=NNLayer.T_Hidden)
l3 = NNLayer(1,activation=NNLayer.A_Linear, layerType=NNLayer.T_Output)
c1 = NNConnection(l1,l2)
c2 = NNConnection(l2,l3)
#c1 = NNConnection(l1,l3)
nnModel = NNModel()
nnModel.addConnection(c1)
nnModel.addConnection(c2)

"""
Create the test data using the number of nodes in the InputLayer for the col
dimension of the X input
"""
# test data
m = 100
n = l1.getNodeCnt()
x = rng.rand(m,n)
# the larger
w = np.linspace(0,3,n) + 1.0
b = 3.5
#err = rng.randn(m)
y = np.asmatrix(np.dot(x,np.transpose(w)) + b).reshape(m,1)

"""
Initialize the model hyper-parameters and run the model to get the resultant
gradients for each model parameter
"""
alpha = 0.0001
# WARNING: Regularization strength will impact the values, so it needs to be
# very low or set to zero
regStrength = 0.01

# get initial copy of model params
params = nnModel.getModelParams()
c_params = deepcopy(params)

# Run the model through one iteration forward and backwards
nnModel.forwardPropigation(x,y)
nnModel.backPropigation(y,alpha,regStrength)
grads = nnModel.getModelGradients()

"""
Test the backpropigation using the epsilon value
"""
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
                nnModel.setModelParams(params)
                nnModel.forwardPropigation(x,y)
                nnModel.backPropigation(y,alpha,regStrength)
                costPlus = nnModel.getCost(y, regStrength)
                
               # update the parameter
                params = deepcopy(c_params)
                params[l][obj][row,col] -= eps
                # run the model and get cost
                nnModel.setModelParams(params)
                nnModel.forwardPropigation(x,y)
                nnModel.backPropigation(y,alpha,regStrength)
                costMinus = nnModel.getCost(y, regStrength)
                
                # get gradient
                grad_num = grads[l][obj][row,col]
                grad_res = (costPlus - costMinus)/(2*eps)
                #
                
                # Raise error if the numerical grade is not close to the backprop gradient
                if not np.isclose(grad_num, grad_res, atol=1e-05):
                    raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_res)))    
                else:
                    if (obj == 0):
                        print "Layer:%d Weights(%d,%d): %3.6f = %3.6f" % (l,row,col,grad_num, grad_res)
                    else:
                        print "Layer:%d Bias(%d,%d): %3.6f = %3.6f" % (l,row,col,grad_num, grad_res)
        
        
        
    