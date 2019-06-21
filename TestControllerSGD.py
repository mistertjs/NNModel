# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:36:55 2016

@author: tschlosser
"""
import numpy as np
rng = np.random
from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from NNController import NNController
import NNFunctions as nnf
import NNStatFunctions as nns
from sklearn import datasets 
import matplotlib.pyplot as plt

m = 200
n_outliers = 10
n = 5

# create sample data set, where 'noise' keeps the 'cost' of the NN model
# from reaching zero and the cost baseline tends to be related to the noise
# value
x, y, coef = datasets.make_regression(n_samples=m, n_features=n,
                                      n_informative=1, noise=20,
                                      coef=True, random_state=None)
y = np.asmatrix(y).reshape(m,1)                                      
                                      
# Always make sure the input and output layers and their connections
# are the first and last in the list
l1 = NNLayer(n,activation=NNLayer.A_Linear, layerType=NNLayer.T_Input)
l2 = NNLayer(3,activation=NNLayer.A_Linear, layerType=NNLayer.T_Hidden)
l3 = NNLayer(1,activation=NNLayer.A_Linear, layerType=NNLayer.T_Output)
c1 = NNConnection(l1,l2)
c2 = NNConnection(l2,l3)
#c1 = NNConnection(l1,l3)
nnModel = NNModel()
nnModel.addConnection(c1)
nnModel.addConnection(c2)


# Build the controller and train the model
nnController = NNController(nnModel)
for i in range(20):
    results = nnController.trainModelSGD(x, y, maxRuns=10, minError=0.05, 
                                      alpha = 0.001, regStrength=0.01,
                                      cvRatio=0.3, printRun=True)
numRuns = results[0]
print "Number of iterations: %d" % numRuns
nnController.plotTraining()

yh = nnModel.nnOutput(x)
w  = nns.leastSquares(x, y)
wh = nns.leastSquares(x,yh)
dw = wh - w
print "W delta: %f" % np.sqrt(np.dot(dw.T,dw))
print "Correlation: %f" % nns.cor(y,yh)
plt.plot(y,yh, 'r.')
