# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 08:52:04 2016

@author: Administrator
"""
import numpy as np
import NNStatFunctions as nns
rng = np.random


'''
Create a linear data set with specific characteristics

Input provides a covariance matrix to give special relationship in the data
https://www.r-bloggers.com/simulating-data-following-a-given-covariance-structure/

'''       
def getLinearData(m=100, n=3, w=None, b=None, cvm=None):
    # Test data
    x = rng.rand(m,n)
    # assumes Covariance Matrix is square (nxn) and correctly populated
    if (cvm is not None):
        L = np.linalg.cholesky(cvm)
        x = np.dot(rng.normal(size=(m,n)), L.T)
        
    # randomize weights and bias
    if (w is None):
        w = np.linspace(0,3,n) + 1.0
    if (b is None):
        b = rng.rand()
    #err = rng.randn(m)
    y = np.asmatrix(np.dot(x,np.transpose(w)) + b).reshape(m,1)        
    
    data = [m, n, w, b, x, y]
    
    # get stats information
    uX = np.mean(x, axis=0, keepdims=True)
    vX = np.var(x, axis=0, keepdims=True)
    uY = np.mean(y, axis=0, keepdims=True)
    vY = np.var(y, axis=0, keepdims=True)
    covXX = nns.cov(x,x)
    covXY = nns.cov(x,y,)
    stats = [uX,vX,uY,vY,covXX,covXY]

    # return data list
    return ['Linear', data, stats]
    
def showLinearStats(stats):
    print "Avg X: %s" % stats[0]
    print "Var X: %s" % stats[1]
    
    


