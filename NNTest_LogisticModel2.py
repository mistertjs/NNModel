# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:33:53 2016

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
import matplotlib.pyplot as plt
rng = np.random
from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from NNController import NNController

# Plot the scatter
doPlot = True
# Set this to True to retain weights for subsequent runs and False to 
# initialize each time
retainInitialWeights = True
doInitialize = True

# Set true to test gradients
doGradientTest = True

nnModel = None
nnParams = None

'''
Start from here to keep initial model and training data. Change hyper params
to evaluate differences in runs
'''
# Always make sure the input and output layers and their connections
# are the first and last in the list
if (doInitialize):
    lIn = NNLayer(2,activation=NNLayer.A_Tanh, layerType=NNLayer.T_Input)
    lH1 = NNLayer(10,activation=NNLayer.A_ReLU, layerType=NNLayer.T_Hidden)
    lH2 = NNLayer(10,activation=NNLayer.A_ReLU, layerType=NNLayer.T_Hidden)
    lOut = NNLayer(1,activation=NNLayer.A_Tanh, layerType=NNLayer.T_Output)
    c1 = NNConnection(lIn,lH1)
    c2 = NNConnection(lH1,lH2)
    c3 = NNConnection(lH2,lOut)
    #c1 = NNConnection(l1,l3)
    nnModel = NNModel()
    nnModel.addConnection(c1)
    nnModel.addConnection(c2)
    nnModel.addConnection(c3)
    
    # copy
    if (retainInitialWeights):
        nnParams = nnModel.getModelParams()
else:
    nnModel.setModelParams(nnParams)
    
# Test data
if (doInitialize):
    m = 400
    n = lIn.getNodeCnt()
    x = 10*(rng.rand(m,n) - 0.5)
    
    # the larger
    w = np.linspace(-6,6,n) + 1.0
    b = -3.5
    err = 10*rng.randn(m)
    z = np.asmatrix(np.dot(x,np.transpose(w)) + b + err).reshape(m,1)
    # logistic function
    ym = 1/(1 + np.exp(-z))
    # binary output
    y  = (ym >= 0.5) + 0.0
    
# Clear initialize flag to keep both model and data
if (retainInitialWeights):
    doInitialize = False
    
# Build the controller and train the model
nnController = NNController(nnModel)
results = nnController.trainModel(x, y, maxRuns=10000, minError=0.0005, 
                                  alpha = 0.015, regStrength=0.001,
                                  gradientCheckData=None, 
                                  printRun=False)

# get output
yh = np.around(nnController.getOutput())

# Show the results
numRuns = results[0]
print "Runs to converge: %d" % numRuns
cost = results[1]
print "Cost: %f" % cost

# show the input averages
xAvg = np.mean(x, axis=0)
xAvg = np.sum(np.dot(xAvg.T,xAvg))
print "Y Avg: %3.3f, X Avg: %3.3f" % (np.mean(y), xAvg)

# show the gradient trail for each weight
nnController.plotTraining()

if (doPlot):
    # Plot both classes on the x1, x2 plane
    #plt.figure(figsize=(10, 15), dpi=100)
    # get classification from original function
    x_red_y = x[np.where(y==1)[0],:]
    x_blue_y = x[np.where(y==0)[0],:]
    # get classifications from estimated function
    x_red_yh = x[np.where(yh==1)[0],:]
    x_blue_yh = x[np.where(yh==0)[0],:]
    #plt.subplot(4,1,1)
    plt.scatter(x_red_y[:,0], x_red_y[:,1], s=40, color='r', marker='s')
    plt.scatter(x_blue_y[:,0], x_blue_y[:,1], s=40, color='b', marker='s')
    plt.scatter(x_red_yh[:,0], x_red_yh[:,1], color='k', marker='o')
    plt.scatter(x_blue_yh[:,0], x_blue_yh[:,1], color='y', marker='o')
    plt.axis([-6, 6, -6, 6])
    plt.grid()
    plt.show()

# output error
tP = np.sum(y[np.where(yh==1)[0]])
fN = np.sum(y[np.where(yh==0)[0]])
aP = np.sum(y)
aN = len(y) - aP
eP = np.sum(yh)
eN = len(yh) - eP
fP = eP - tP
tN = eN - fN
accuracy = (tN + tP)/len(y)
error = 1.0 - accuracy

print "Error Out: %2.5f" % error

                 