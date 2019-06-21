# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 06:13:52 2016

@author: Administrator
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
rng = np.random

def cov(x, y, doPopulation=True):
    ux = np.mean(x, axis=0)
    uy = np.mean(y, axis=0)
    vx = x - ux
    vy = y - uy
    m = np.shape(x)[0]
    if (doPopulation == False):
        m = m - 1
    return np.dot(vx.T, vy) / m
    
def cor(x, y=None, doPopulation=True):
    if (y is not None):
        xyCov = cov(x,y,doPopulation)
        stdX = np.std(x, axis=0)
        stdY = np.std(y, axis=0)
        return np.divide(xyCov, np.multiply(stdX, stdY))
    else:
        return np.corrcoef(x.T)
        
def var(x, doPopulation=True):
    return np.diagonal(cov(x,x,doPopulation))

def leastSquares(X, Y):
    """
    Return W as a function of X and Y where X is a multivariate matrix
    and Y is a single column matrix of a linear regression output
    """
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)