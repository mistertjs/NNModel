# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:33:53 2016

This example is the simplest model with no hidden layer to compare the weights
with the model weights and bias.
If regularization is used, the model will not converge as exactly as it could,
but of course it will not overfit
Adding any random error will cause variance in the weights

@author: Administrator
"""

import numpy as np
rng = np.random
from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from NNController import NNController


# Always make sure the input and output layers and their connections
# are the first and last in the list
lIn = NNLayer(3, layerType=NNLayer.T_Input)
lOut = NNLayer(1,activation=NNLayer.A_Linear, layerType=NNLayer.T_Output)
c1 = NNConnection(lIn,lOut)
#c1 = NNConnection(l1,l3)
nnModel = NNModel()
nnModel.addConnection(c1)

# Test data
m = 300
n = lIn.getNodeCnt()
x = rng.rand(m,n)
# the larger
w = np.linspace(0,3,n) + 1.0
b = 3.5
#err = rng.randn(m)
y = np.asmatrix(np.dot(x,np.transpose(w)) + b).reshape(m,1)


# Build the controller and train the model
nnController = NNController(nnModel)
results = nnController.trainModel(x, y, maxRuns=10000, minError=0, 
                                  alpha = 0.01, regStrength=0,
                                  printRun=False)

yh = nnModel.nnOutput(x)

# compare weights and bias
print w,b
nnModel.showWeights()

# Show the results
numRuns = results[0]
print "Runs to converge: %d" % numRuns
cost = results[1]
print "Cost: %f" % cost

# show the input averages
xAvg = np.mean(x, axis=0)
xAvg = np.sum(np.dot(xAvg.T,xAvg))
print "Y Avg: %3.3f, X Avg: %3.3f" % (np.mean(y), xAvg)

nnController.plotTraining()


