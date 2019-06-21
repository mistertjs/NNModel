"""
Created on Fri Aug 12 05:33:53 2016

Runs a training set using Softmax output layer and the Iris dataset

@author: Administrator
"""

from sklearn import datasets

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from NNController import NNController

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
lH1 = NNLayer(20,activation=NNLayer.A_Tanh, layerType=NNLayer.T_Hidden)
lOut = NNLayer(numClasses,activation=NNLayer.A_Softmax, layerType=NNLayer.T_Output)
c1 = NNConnection(lIn,lH1)
c2 = NNConnection(lH1,lOut)

#c1 = NNConnection(l1,l3)
nnModel = NNModel()
nnModel.addConnection(c1)
nnModel.addConnection(c2)

# Build the controller and train the model
nnController = NNController(nnModel)

#yh = nnModel.forwardPropigation(X, Y)
#cost = nnModel.getCost(Y)


# Train the model
results = nnController.trainModel(X, Y, maxRuns=2500, minError=None,
                   alpha=0.001, regStrength=0.001,
                   gradientCheckData=None, printRun=False,
                   sgdRatio=0.0)
print "Runs: %d" % results[0]
print "Cost: %f" % results[1]

nnController.plotTraining()
yh = (nnModel.nnOutput(X) >= 0.5).astype(int)

predicted_class = yh
print 'training accuracy: %.2f' % (np.mean(predicted_class == Y))  