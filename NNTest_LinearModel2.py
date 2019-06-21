# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:33:53 2016

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
lIn = NNLayer(3,activation=NNLayer.A_Linear, layerType=NNLayer.T_Input)
lH1 = NNLayer(10,activation=NNLayer.A_Linear, layerType=NNLayer.T_Hidden)
lOut = NNLayer(1,activation=NNLayer.A_Linear, layerType=NNLayer.T_Output)
c1 = NNConnection(lIn,lH1)
c2 = NNConnection(lH1,lOut)
#c1 = NNConnection(l1,l3)
nnModel = NNModel()
nnModel.addConnection(c1)
nnModel.addConnection(c2)

# Test data
m = 100
n = lIn.getNodeCnt()
x = rng.rand(m,n)
# the larger
w = np.linspace(0,3,n) + 1.0
b = 3.5
err = rng.randn(m)
y = np.asmatrix(np.dot(x,np.transpose(w)) + b + err).reshape(m,1)


# Build the controller and train the model
nnController = NNController(nnModel)
results = nnController.trainModel(x, y, maxRuns=2000, minError=None, 
                                  alpha = 0.001, regStrength=0.01,
                                  gradientCheckData=None, 
                                  printRun=False)

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
