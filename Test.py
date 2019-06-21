# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 06:30:48 2016

@author: Administrator
"""

#import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
#sys.path.append('C:/Users/Administrator/Google Drive/Python\NNModel')

import numpy as np
import matplotlib.pyplot as plt
rng = np.random
from NNLayer import NNLayer
from NNConnection import NNConnection
from NNModel import NNModel
from copy import deepcopy

l1 = NNLayer(3,activation=NNLayer.A_Linear, layerType=NNLayer.T_Input)
l2 = NNLayer(2,activation=NNLayer.A_Linear, layerType=NNLayer.T_Hidden)
l3 = NNLayer(1,activation=NNLayer.A_Linear, layerType=NNLayer.T_Output)
c1 = NNConnection(l1,l2)
c2 = NNConnection(l2,l3)
#c1 = NNConnection(l1,l3)
nnModel = NNModel()
nnModel.addConnection(c1)
nnModel.addConnection(c2)

params = nnModel.getModelParams()
paramLen = len(params)
for l in range(paramLen):
    p = params[l]
    W = p[0]
    b = p[1]
    print "###Layer: %d \nWeights: \n%s \nBias: \n%s" % (l, np.around(W,2), np.around(b,2))

c_params = deepcopy(params)

# insert new Weight and Bias in connection 2
wS = np.shape(params[1][0])
W = np.identity(wS[0])
params[1][0] = W
params[1][1] = np.ones((1,1)) * 0.5
nnModel.setModelParams(params)

params = nnModel.getModelParams()
paramLen = len(params)
for l in range(paramLen):
    p = params[l]
    W = p[0]
    b = p[1]
    print "###Layer: %d \nWeights: \n%s \nBias: \n%s" % (l, np.around(W,2), np.around(b,2))