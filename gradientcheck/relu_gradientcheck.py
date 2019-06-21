# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 06:30:48 2016

Performs gradient check on ReLU hidden layer network using dataset of two 
classes.

To bypass gradient check and test for convergence, set bCheckConverge = True

Expectation is for gradient check to fail as gradients get smaller

Read the following to understand how to determine if the difference is
significant, particularly the paragraph 'Kinks in the objective' to show
how ReLU functions are problmatic when evaluating (f+h), then (f-h)
http://cs231n.github.io/neural-networks-3/

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
lH1 = NNLayer(4, activation=NNLayer.A_ReLU, layerType=NNLayer.T_Hidden)
lOut = NNLayer(1, activation=NNLayer.A_Tanh, layerType=NNLayer.T_Output)
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
m = 10
n = lIn.getNodeCnt()
x, yc = make_classification(n_samples=m, n_features=n, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1)
y = np.matrix(yc, dtype='float64').reshape(m,1)
x = np.matrix(x, dtype='float64').reshape(m,n)

# mean substraction for preprocessing
x -= np.mean(x, axis = 0)
#y -= np.mean(y)
y[y==0] = -1

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
maxRuns = 100
if (bCheckConverge):
    maxRuns = 10000
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
        eps = 01e-5
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

                            # get relative error
                            fa = grad_num; fn = grad_res
                            relError = np.abs(fa - fn)/np.max(np.abs(fa),np.abs(fn))

                            # Raise error if the numerical grade is not close to the backprop gradient
                            #if not bBreakAll and relError >= 1e-04: 
                            if not bBreakAll and not np.isclose(grad_num, grad_res, atol=1e-05):
                                errorCnt += 1
                                print('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_res)))    
                                #bBreakAll = True
                                if (obj == 0):
                                    print "Layer:%d Weights(%d,%d): %3.6f = %3.6f" % (l,row,col,grad_num, grad_res)
                                else:
                                    print "Layer:%d Bias(%d,%d): %3.6f = %3.6f" % (l,row,col,grad_num, grad_res)

# show results from convergence or gradient test        
if bCheckConverge:
    plt.subplot(212)
    plt.plot(costVals, 'r.')
else:    
    print "Total Errors: %d" % errorCnt        
    
yh = nnModel.nnOutput(x)
yo = np.around(yh)
errOut = np.sum(np.abs(yo - y))/m
accuracy = 1.0 - errOut
print "Single Layer Accuracy: %2.5f" % accuracy    