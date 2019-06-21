# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 06:30:48 2016

Performs gradient check on Sigmoid network using dataset of two classes.
To bypass gradient check and test for convergence, set bCheckConverge = True

Expectation is for gradient check to fail as gradients get smaller

@author: Administrator
"""

import os
import sys
compName = os.environ['COMPUTERNAME']

# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
default_path = 'C:/Users/Administrator/Google Drive/Python/NNModel'
if (compName == "GAC2TSCHLOS"):
    default_path = 'C:/Users/tschlosser/Google Drive/Python/NNModel'

# set path and current directory
sys.path.append(default_path)

import numpy as np
rng = np.random

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from copy import deepcopy

# set this true to bypass gradient check and test for convergence
bCheckConverge = False

"""
Define the model structure
"""
lIn = NNLayer(2, layerType=NNLayer.T_Input)
lH1 = NNLayer(10, activation=NNLayer.A_Sigmoid, layerType=NNLayer.T_Hidden)
lOut = NNLayer(1, activation=NNLayer.A_Sigmoid, layerType=NNLayer.T_Output)
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
n = lIn.getNodeCnt()
x, yc = make_classification(n_features=n, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1)
m = len(yc)
# generate matrix with float64 necessary for preprocessing mean
y = np.matrix(yc, dtype='float64').reshape(m,1)

# mean substraction for preprocessing
#x -= np.mean(x, axis = 0)
#y -= np.mean(y)

plt.subplot(211)
plt.scatter(x[:, 0], x[:, 1], marker='o', c=yc)
    
"""
Initialize the model hyper-parameters and run the model to get the resultant
gradients for each model parameter
"""
alpha = 01e-2
# WARNING: Regularization strength will impact the values, so it needs to be
# very low or set to zero
regStrength = 01e-3

# set max runs to gradient check or converge
maxRuns = 500
if (bCheckConverge):
    maxRuns = 1000
    alpha *= 10.
printInterval = maxRuns / 10

# track errors
errorCnt = 0
bBreakAll = False
costVals = np.zeros(maxRuns)
for curRun in range(maxRuns):

    # get initial copy of model params
    params = nnModel.getModelParams()
    c_params = deepcopy(params)
    
    # Run the model through one iteration forward and backwards
    nnModel.forwardPropigation(x,y)
    nnModel.backPropigation(y,alpha,regStrength)
    
    # print run and cost every 10 runs
    cost = nnModel.getCost(y)
    costVals[curRun] = cost
    if curRun % printInterval == 0:
        print "iteration %d: loss %f" % (curRun, cost)     
    """
    Test the backpropigation using the epsilon value
    """
    grads = nnModel.getModelGradients()
    
    if not bCheckConverge:
        eps = 01e-4
        # iterate through each model parameter and compare its gradient with cost delta          
        if bBreakAll:
            break;    
        else:
            for l in range(len(c_params)):
                # get param list for this connection with 2 objects W,b
                if bBreakAll:
                    break;
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
                            if not bBreakAll and not np.isclose(grad_num, grad_res, atol=1e-05):
                                errorCnt += 1
                                print('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_res)))    
                                bBreakAll = True

# show results from convergence or gradient test        
if bCheckConverge:
    plt.subplot(212)
    plt.plot(costVals, 'r.')

print "Total Gradient Errors: %d" % errorCnt        

# get model output
yo = nnModel.nnOutput(x)
yo = np.around(yo)
# check accuracy 
errorRatio = np.sum((yo - y) <> 0) / float(len(y))
print "Accuracy: %2.5f" % (1 - errorRatio)
    