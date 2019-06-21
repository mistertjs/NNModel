# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 06:30:48 2016

@author: Administrator
"""

#import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
#sys.path.append('C:/Users/Administrator/Google Drive/Python\NNModel')

from sklearn import datasets

import sys
import os
import numpy as np
import pandas as pd
rng = np.random.RandomState(23455)
compName = os.environ['COMPUTERNAME']

# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
default_path = 'C:/Users/Administrator/Google Drive/Python/NNModel'
if (compName == "GAC2TSCHLOS"):
    default_path = 'C:/Users/tschlosser/Google Drive/Python/NNModel'

# set path and current directory
sys.path.append(default_path)

from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from copy import deepcopy

"""
Define the model structure
"""
# import some data to play with
iris = datasets.load_iris()
X = iris.data
Y = iris.target
Yd = pd.get_dummies(Y)
Y = Yd.as_matrix()

# Test data
m = X.shape[0]
n = X.shape[1]
numClasses = Y.shape[1]

# Always make sure the input and output layers and their connections
# are the first and last in the list
lIn = NNLayer(n,activation=NNLayer.A_Tanh, layerType=NNLayer.T_Input)
lH1 = NNLayer(10,activation=NNLayer.A_Tanh, layerType=NNLayer.T_Hidden)
lOut = NNLayer(numClasses,activation=NNLayer.A_Softmax, layerType=NNLayer.T_Output)
c1 = NNConnection(lIn,lH1)
c2 = NNConnection(lH1,lOut)

#c1 = NNConnection(l1,l3)
nnModel = NNModel()
nnModel.addConnection(c1)
nnModel.addConnection(c2)

"""
Create the test data using the number of nodes in the InputLayer for the col
dimension of the X input
"""


"""
Initialize the model hyper-parameters and run the model to get the resultant
gradients for each model parameter
"""
alpha = 0.0001
# WARNING: Regularization strength will impact the values, so it needs to be
# very low or set to zero
regStrength = 0

# get initial copy of model params
params = nnModel.getModelParams()
c_params = deepcopy(params)

# Run the model through one iteration forward and backwards
out = nnModel.forwardPropigation(X,Y)

nnModel.backPropigation(Y,alpha,regStrength)
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
                nnModel.forwardPropigation(X,Y)
                nnModel.backPropigation(Y,alpha,regStrength)
                costPlus = nnModel.getCost(Y)
                
               # update the parameter
                params = deepcopy(c_params)
                params[l][obj][row,col] -= eps
                # run the model and get cost
                nnModel.setModelParams(params)
                nnModel.forwardPropigation(X,Y)
                nnModel.backPropigation(Y,alpha,regStrength)
                costMinus = nnModel.getCost(Y)
                
                # get gradient
                if (p.shape[0] > 1):
                    grad_num = grads[l][obj][row,col]
                else:
                    # grads is a vector, not a matrix
                    grad_num = grads[l][obj][col]
                    
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
        
        
        
T = Y
Y = out
E = np.multiply(Y,T-1.) + np.multiply(Y,T) - T
'''
for i in range(m):
    v = Y[i];
    t = T[i]
    # get diag of p(1-p)
    vd = np.multiply(np.diag(v), 1-v)
    # get product of values
    vp = -np.dot(v.reshape(3,1),v.reshape(1,3))
    # fill vp with vd diagonal
    np.fill_diagonal(vp,np.diag(vd))
    # get Y-T product
    yc = np.dot(vp, -np.divide(t,v))
'''    