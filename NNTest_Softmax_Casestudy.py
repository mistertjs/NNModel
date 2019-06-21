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
rng = np.random.RandomState(23455)
compName = os.environ['COMPUTERNAME']

# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
default_path = 'C:/Users/Administrator/Google Drive/Python/NNModel'
if (compName == "GAC2TSCHLOS"):
    default_path = 'C:/Users/tschlosser/Google Drive/Python/NNModel'

# set path and current directory
sys.path.append(default_path)

# get data path
dataPath = 'C:/Users/Administrator/Google Drive/Python/data/'
if (compName == "GAC2TSCHLOS"):
    dataPath = 'C:/Users/tschlosser/Google Drive/Python/data/'

from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from NNController import NNController

'''
Choose the data to be used. If existing data, the results should
be compared to the softmax-case-study.py under Python/working
'''
bUseExistingData = True
if (bUseExistingData):
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classe    
    # load X from saved dataset
    X = np.genfromtxt(
        dataPath + 'softmax.csv',  # file name
        skip_header=0,          # lines to skip at the top
        skip_footer=0,          # lines to skip at the bottom
        delimiter=',',          # column delimiter
        #dtype='float32',        # data type
        filling_values=0)       # fill missing values with 0
      
    # create class labels
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in xrange(K):        
        ix = range(N*j,N*(j+1))
        y[ix] = j

    # get dummies from single vector
    Yd = pd.get_dummies(y)
    Y = Yd.as_matrix()
        
else:        
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
lIn = NNLayer(n,activation=NNLayer.A_Linear, layerType=NNLayer.T_Input)
lH1 = NNLayer(100,activation=NNLayer.A_ReLU, layerType=NNLayer.T_Hidden)
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
step_size = 1e-1
reg = 1e-3 # regularization strength
results = nnController.trainModel(X, Y, maxRuns=10000, minError=None,
                   alpha=step_size, regStrength=reg,
                   gradientCheckData=None, printRun=False,
                   sgdRatio=0.0)
print "Runs: %d" % results[0]
print "Cost: %f" % results[1]

nnController.plotTraining()
yh = (nnModel.nnOutput(X) >= 0.5).astype(int)

predicted_class = yh
print 'training accuracy: %.2f' % (np.mean(predicted_class == Y))  