# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:36:55 2016

@author: tschlosser
"""
import numpy as np
rng = np.random
import datetime as dt
from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from NNController import NNController
import NNFunctions as nnf
import NNStatFunctions as nns
from sklearn import datasets 
import matplotlib.pyplot as plt

m = 2000000
n_outliers = m/10
n = 5

# create sample data set, where 'noise' keeps the 'cost' of the NN model
# from reaching zero and the cost baseline tends to be related to the noise
# value
x, y, coef = datasets.make_regression(n_samples=m, n_features=n,
                                      n_informative=1, noise=10,
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

# save initial params
nnParams = nnModel.getModelParams()
nnModel.setModelParams(nnParams)

'''
Run SG
'''
nnController = NNController(nnModel)
begTime = dt.datetime.now()
results = nnController.trainModelSGD(x, y, maxRuns=200, minError=0.05, 
                                  alpha = 0.001, regStrength=0.01,
                                  cvRatio=0.2,
                                  printRun=False)
endTime = dt.datetime.now()                                  
difTime = (endTime - begTime).total_seconds()
numRuns = results[0]
yRows = results[4]
print "Number of iterations: %d, Length Y: %d, Time: %2.3f" % (numRuns,yRows,difTime)
yh = nnModel.nnOutput(x)
w  = nns.leastSquares(x, y)
wh = nns.leastSquares(x,yh)
dw = wh - w
print "W delta: %f" % np.sqrt(np.dot(dw.T,dw))
print "Correlation: %f" % nns.cor(y,yh)
#nnController.plotTraining()


# Build the controller and train the model
nnModel.setModelParams(nnParams)
nnController = NNController(nnModel)
lenY = 0
begTime = dt.datetime.now()
for i in range(200):
    results = nnController.trainModelSGD(x, y, maxRuns=1, minError=0.05, 
                                      alpha = 0.001, regStrength=0.01,
                                      batchRatio=0.01,
                                      cvRatio=0.5, printRun=False)
    xRows = results[3]                                      
    yRows = results[4]
    bBreakEarly = results[5]                                      
    if(bBreakEarly):
        break
    '''
    numRuns = results[0]
    cost = results[1]
    lenY = results[3]
    yh = nnModel.nnOutput(x)
    if (nns.cor(y,yh) >= 0.95):
        print "Cost:%f" % cost
        break
    '''
endTime = dt.datetime.now()                                  
difTime = (endTime - begTime).total_seconds()
print "Number of iterations: %d, Length Y: %d, Time: %2.3f" % (i,yRows,difTime)

#nnController.plotTraining()

yh = nnModel.nnOutput(x)
w  = nns.leastSquares(x, y)
wh = nns.leastSquares(x,yh)
dw = wh - w
print "W delta: %f" % np.sqrt(np.dot(dw.T,dw))
print "Correlation: %f" % nns.cor(y,yh)
#plt.plot(y,yh, 'r.')
