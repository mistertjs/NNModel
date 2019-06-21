# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 06:30:48 2016

This file allows you to check the gradient very specifically on a the first
weight component. End prints out whether there is a difference
@author: Administrator
"""

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('C:/Users/Administrator/Google Drive/Python\NNModel')

import numpy as np
rng = np.random
import matplotlib.pyplot as plt
import NNFunctions as nnf

# Test data
m = 200
n = 3
x = 10*(rng.rand(m,n) - 0.5)

# the larger
w = np.linspace(-6,6,n) + 1.0
b = -1.5
err = 10*rng.randn(m)
z = np.asmatrix(np.dot(x,np.transpose(w)) + b + err).reshape(m,1)
# logistic function
ym = 1/(1 + np.exp(-z))
# binary output
T  = (ym >= 0.5) + 0.0

# random weights to simulate nn model
W = np.asmatrix([(0.1),(-0.2),(0.3)]).reshape(3,1)
bh = 0.5

# forward propigate
Y = nnf.tanhActivation(x,W,bh)

# back propigate to get gradients
cDer = np.divide((Y - T), (1 - np.square(Y)))
#cDer = np.divide(np.multiply(Y,T) - 1, 1 - np.square(Y))
yDer = nnf.tanhDerivative(Y)
g_W = np.dot(x.T, np.multiply(cDer, yDer))/m

# same as g_W
g_W_test = np.dot(x.T, (Y - T))/m

for i in range(len(W)):
    # increment W[0]
    eps = 0.0001
    Wp = np.copy(W); Wp[i] += eps
    Y = nnf.tanhActivation(x,Wp,bh)
    costPlus = nnf.costTanh(Y, T)
    
    Wn = np.copy(W); Wn[i] -= eps
    Y = nnf.tanhActivation(x,Wn,bh)
    costMinus = nnf.costTanh(Y, T)
    
    d_W = (costPlus - costMinus)/(2*eps)
    print "Ratio BP: %f" % (g_W[i]/d_W)
    
    # check for close values
    if not np.isclose(d_W, g_W[i], atol=1e-05):
        print "Error in values %f" % (d_W - g_W[i])
    