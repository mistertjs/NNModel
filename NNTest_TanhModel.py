# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:33:53 2016

INTERESTING: A mistake cause by using nnf.costLinear(Y,T) in the 
NNLayer.getCost() function created a much better model fit than using
the nnf.costTanh function. Check into this as to why that might be better.
Its known that the 'full' derivative of this cost function is (Y-T), which
is similar to the 'full' linear model cost function

@author: Administrator
"""
#import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
#sys.path.append('C:/Users/Administrator/Google Drive/Python/NNModel')

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets # To generate the dataset
rng = np.random
from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from NNController import NNController
#from NNTestData import NNTestData

# Plot the scatter
doPlot = True

# Always make sure the input and output layers and their connections
# are the first and last in the list
lIn = NNLayer(2,activation=NNLayer.A_Tanh, layerType=NNLayer.T_Input)
lH1 = NNLayer(4,activation=NNLayer.A_Tanh, layerType=NNLayer.T_Hidden)
lH2 = NNLayer(3,activation=NNLayer.A_Tanh, layerType=NNLayer.T_Hidden)
lOut = NNLayer(1,activation=NNLayer.A_Tanh, layerType=NNLayer.T_Output)
c1 = NNConnection(lIn,lH1)
c2 = NNConnection(lH1,lH2)
c3 = NNConnection(lH2,lOut)
#c1 = NNConnection(l1,l3)
nnModel = NNModel()
nnModel.addConnection(c1)
nnModel.addConnection(c2)
nnModel.addConnection(c3)

# store initial parameters
if 'nnParams' not in locals():
    nnParams = nnModel.getModelParams()
else:
    nnModel.setModelParams(nnParams)

# Test data
m = 100
x, y = sklearn.datasets.make_circles(n_samples=m, shuffle=False, factor=0.3, noise=0.1)
Y = np.zeros((100,2)) # Define target matrix
Y[y==1,1] = 1
Y[y==0,0] = -1

n = lIn.getNodeCnt()
y = np.matrix(y).reshape(m,1)

# figure out the weight and bias
w = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y).T
b = np.mean(Y - y, axis=0)

# Build the controller and train the model
nnController = NNController(nnModel)


# set up test data for gradient training
nnTestData = None #NNTestData(m,n,x,w,b,y)

# Train the model
results = nnController.trainModel(x, y, maxRuns=2000, minError=0.005, 
                                  alpha = 0.015, regStrength=0.001,
                                  gradientCheckData=nnTestData, 
                                  printRun=False,
                                  sgdRatio=0.0)
# check for bad gradient test
if (results[0] > 0):
        
        
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
    
    '''
    Check the costRatio curve
    '''
    #plt.plot(nnController.crv, 'r-')
    
    # show the gradient trail for each weight
    nnController.plotTraining()
    
    if (doPlot):
        # Plot both classes on the x1, x2 plane
        #plt.figure(figsize=(10, 15))
        # get classification from original function
        x_red_y = x[np.where(y==1)[0],:]
        x_blue_y = x[np.where(y==0)[0],:]
        # get classifications from estimated function
        x_red_yh = x[np.where(yh==1)[0],:]
        x_blue_yh = x[np.where(yh==0)[0],:]
        #plt.subplot(4,1,1)
        plt.scatter(x_red_y[:,0], x_red_y[:,1], s=120, color='y', marker='o')
        plt.scatter(x_blue_y[:,0], x_blue_y[:,1], s=120, color='r', marker='o')
        plt.scatter(x_red_yh[:,0], x_red_yh[:,1], s=80, color='k', marker='o')
        plt.scatter(x_blue_yh[:,0], x_blue_yh[:,1], s=80, color='b', marker='o')
        plt.axis([-2, 2, -2, 2])
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

'''
# Get the g_W output layer trend
gv = nnModel.getGWVar()
m = len(gv)
gv = np.asarray(gv)
gv = gv.reshape(m,3,1)
plt.plot(gv[:,1,0], 'r.')
'''