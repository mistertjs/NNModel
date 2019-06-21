# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:33:53 2016

This example uses the Sigmoid to model the input. What's interesting is that
there can be a number of solutions because of the probablistic function used
to round a value to either 0 or 1

@author: Administrator
"""
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('C:/Users/tschlosser/Google Drive/Python/NNModel')

import numpy as np
import matplotlib.pyplot as plt
rng = np.random
from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from NNController import NNController
import NNFunctions as nnf

# Always make sure the input and output layers and their connections
# are the first and last in the list
lIn = NNLayer(2, layerType=NNLayer.T_Input)
lOut = NNLayer(1,activation=NNLayer.A_Sigmoid, layerType=NNLayer.T_Output)
c1 = NNConnection(lIn,lOut)

#c1 = NNConnection(l1,l3)
nnModel = NNModel()
nnModel.addConnection(c1)

# Test data
m = 200
n = lIn.getNodeCnt()
x = 10*(rng.rand(m,n) - 0.5)

# the larger
w = np.linspace(-6,6,n) + 1.0
b = -3.5
err = 0 #10*rng.randn(m)
z = np.asmatrix(np.dot(x,np.transpose(w)) + b + err).reshape(m,1)
# logistic to binary function
y = np.around(nnf.logistic(z))

# Build the controller and train the model
nnController = NNController(nnModel)
results = nnController.trainModel(x, y, maxRuns=10000, minError=0., 
                                  alpha = 0.001, regStrength=0.15,
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

print "In  Weights: %s, Bias: %2.2f" % (w,b)
params = nnModel.getModelParams()
print "Out Weights: %s, Bias: %2.2f" % (params[0][0], params[0][1])

# reconstruct
wo = params[0][0].T
bo = params[0][1]
zo = np.asmatrix(np.dot(x,np.transpose(wo)) + bo).reshape(m,1)
# logistic function
yo = np.around(nnf.logistic(zo))
print "Summed Error: %d" % np.sum(yo - y)

doPlot = False
if (doPlot):
    # Plot both classes on the x1, x2 plane
    plt.figure(figsize=(10, 15), dpi=100)
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

#print "Error Out: %2.5f" % error
print "Accuracy: %2.5f" % accuracy

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="log", penalty="l1", fit_intercept=True)
clf.fit(x,y)
print clf.coef_
print clf.intercept_
# reconstruct
wo = clf.coef_
bo = clf.intercept_
zo = np.asmatrix(np.dot(x,np.transpose(wo)) + bo).reshape(m,1)
# logistic function
yo = np.around(nnf.logistic(zo))
print "Summed Error: %d" % np.sum(yo - y)