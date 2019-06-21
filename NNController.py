# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:33:53 2016

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn import cross_validation
rng = np.random
from NNGradientTest import NNGradientTest
import NNFunctions as nnf
import NNStatFunctions as nns
from NNLayer import NNLayer

bTestGradient = False


class NNController(object):
    def __init__(self, nnModel):
        self.nnModel = nnModel
        self.yh = None
        
    '''
    DONT CALL THIS WHILE USING SGD DURING TRAINING...It will push the whole
    y array into the model and mess up the dimensions
    '''        
    def getOutput(self):
        return self.nnModel.nnOutput(self.x_copy)
        
    # Forward Propigate and get output result
    def forwardPropigate(self, x, y):
        return self.nnModel.forwardPropigation(x, y)        
        
    def trainModel(self, x, y, maxRuns=100, minError=None,
                   alpha=0.001, regStrength=0.001,
                   gradientCheckData=None, printRun=False,
                   sgdRatio=0.0):
                       
        # save entire data set
        self.x_copy = np.copy(x)
        self.y_copy = np.copy(y)
                       
        # initialize variables            
        self.maxRuns = maxRuns
        self.minError = minError
        self.alpha = alpha
        self.regStrength = regStrength
        self.costOut = np.zeros(self.maxRuns)
        
        # store the ratio to get the velocity
        self.crv = np.zeros(self.maxRuns)
        
        '''
        Test gradients of the model
        '''
        if (gradientCheckData is not None):
            gradTest = NNGradientTest(self.nnModel, 
                                      nnTestData=gradientCheckData)
            testPassed = gradTest.runTest(printResults=True)
            if (testPassed == False):
                print "Error in the gradient test!"
                return [0,0,0]

        '''
        Iterate until convergence or max runs
        '''
        m = len(y)
        cost = 0
        prior_cost = None
        costRatio = 100
        costIncreaseCnt = 0
        for curRun in range(self.maxRuns):
            
            # check for stochastic gradient ratio
            self.x = self.x_copy
            self.y = self.y_copy
            if (sgdRatio > 0.0):
                k = int(np.around(m * sgdRatio))
                # create a random array for selection
                idx = np.random.choice(m, k, replace=False)
                # select the data set 
                self.x = self.x_copy[idx]
                self.y = self.y_copy[idx]
            
            # Forward Propigate
            self.yh = self.nnModel.forwardPropigation(self.x, self.y)
            
            # Get Cost
            cost = self.nnModel.getCost(self.y, self.regStrength)
            self.costOut[curRun] = cost
            
            # check for delta ratio of cost
            if (self.minError is not None and prior_cost is not None):
                costRatio = (cost - prior_cost)/prior_cost
                self.crv[curRun] = costRatio
                
                # check for reversal
                if (sgdRatio == 0 and cost > prior_cost):
                    print "Cost %f exceeded prior cost %f" % (cost, prior_cost)
                    # truncate the cost array
                    self.costOut = self.costOut[:curRun]                        
                    # Return list of training results
                    return list([curRun, cost, prior_cost])
                    
                # check output type
                if (self.nnModel.getOutputType() in (NNLayer.A_Sigmoid, NNLayer.A_Tanh)):
                    # get error ratio
                    yout = np.around(self.getOutput())
                    tP = np.multiply(y,yout).sum()/np.sum(y)
                    tN = np.multiply(1-y,1-yout).sum()/np.sum(1-y)
                    err = 1.0 - (tP+tN)/2
                    if (err <= self.minError):
                        print "Classification Error: %f" % err
                        break
                else:
                    # check for basic linear error
                    if (self.minError is not None):
                        if (np.abs(costRatio) < self.minError):
                            print "CostRatio: %f" % np.abs(costRatio)
                            # Tryin increasing the learning rate 3x before 
                            # breaking
                            costIncreaseCnt += 1
                            if (costIncreaseCnt >= 3):
                                break
                            else:
                                # increase learning rate
                                self.alpha *= 2
                                print "Increased alpha to %f" % self.alpha                            

            # save cost from this run            
            prior_cost = cost

            
            # learning rate momentum
            #if (costRatio > -0.01)
            # Print large runs every 1000 
            if (printRun == False) and (curRun % 1000 == 0):
                 print "iteration %d: loss %f" % (curRun, cost)    
                 
            # Backpropigate
            # print "Shape: y:%s, yh:%s" % (np.shape(self.y), np.shape(self.yh))
            self.nnModel.backPropigation(self.y, self.alpha, self.regStrength)
            if (printRun):
                print "Run: %s, Cost: %3.5f, Ratio: %2.5f" % (curRun, cost, costRatio)
                
        # truncate the cost array
        self.costOut = self.costOut[:curRun]                        
        
        # Return list of training results
        return list([curRun, cost, prior_cost])
        
    """
    Train the model using Stochastic Gradient Descent
    """        
    def trainModelSGD(self, x, y, maxRuns=100, minError=None,
                   alpha=0.001, regStrength=0.001, batchRatio=None,
                   cvRatio=0.2, printRun=False):
        
        # save entire data set
        self.x_copy = np.copy(x)
        self.y_copy = np.copy(y)
        
        # create training and test set
        xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(
        x, y, test_size=cvRatio, random_state=None)

        # create minibatch training set
        if (batchRatio is not None):
            now = dt.datetime.now()
            rand_state = int(round((now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds(),0))
            xTrain, xbTest, yTrain, ybTest = cross_validation.train_test_split(
            xTrain, yTrain, test_size=1-batchRatio, random_state=rand_state)
        
        yTrain = np.asmatrix(yTrain).reshape(len(yTrain),1)
        yTest = np.asmatrix(yTest).reshape(len(yTest),1)

        # assign for later plotting
        self.x = xTrain
        self.y = yTrain
        
        # get training size
        xRows = np.shape(xTrain)[0]
        yRows = np.shape(yTrain)[0]

        # show size of set        
        print "Train size: %d" % len(yTrain)
        print "yTrain: %f" % np.sum(yTrain)
        print "Counts: Train:%d, Test:%d" % (len(yTrain), len(yTest))
        
        cost = 0
        bBreakEarly = False
        self.costOut = np.zeros(maxRuns)
        prior_cost = None
        costRatio = 1
        prior_err = None
        errRatio = 1
        alphaAdjustCnt = 0
        for curRun in range(maxRuns):        
            # Forward Propigate
            self.yh = self.nnModel.forwardPropigation(xTrain, yTrain)
            
            # Get Cost
            cost = self.nnModel.getCost(yTrain, regStrength)
            self.costOut[curRun] = cost            

            # Validate and compare error
            cvError = 0
            yhTest = self.nnModel.nnOutput(xTest)
            if (minError is not None):
                # Linear regression hold out test
                if (self.nnModel.getOutputType() == NNLayer.A_Linear):
                    # get the normalized root mean squared error                
                    corrError = 1.0 - np.abs(nns.cor(yhTest, yTest))
                    #if (errRatio is not None and np.abs(errRatio) <= minError):
                    if (errRatio is not None and corrError <= minError):
                        bBreakEarly = True
                        # check if this has been the nth time to adjust
                        #if (alphaAdjustCnt > 5):
                            #print "NRMSD Error Ratio Reached: %f" % cvError
                            #print "Run: %s, Cost: %3.5f, CostRatio: %2.5f Error: %3.5f, ErrorRatio: %3.5f" % (curRun, cost, costRatio, cvError, errRatio)
                        break;
                        '''
                        # start from a new position
                        print "Getting new position for %dnt time" % alphaAdjustCnt
                        xTrain, xTest, yTrain, yTest = self.resplitData(cvRatio)
                        print "yTrain: %f" % np.sum(yTrain)
                        self.yh = self.nnModel.forwardPropigation(xTrain, yTrain)
                        '''
                    if (curRun > 2 and prior_cost is not None and prior_cost < cost):
                        print "Prior cost exceeded cost!"
                        break;                        

            # update cost and error ratio
            if (prior_cost is not None):
                costRatio = (cost - prior_cost)/prior_cost

            # calc cost ratio momentum
            prior_cost = cost                
            prior_err  = cvError

            # Print large runs every 1000 
            if (printRun == False) and (curRun % 1000 == 0):
                 print "iteration %d: loss %f" % (curRun, cost)    

            # Backpropigate
            # print "Shape: y:%s, yh:%s" % (np.shape(self.y), np.shape(self.yh))
            self.nnModel.backPropigation(yTrain, alpha, regStrength)
            if (printRun):
                print "Run: %s, Cost: %3.5f, CostRatio: %2.5f Error: %3.5f, ErrorRatio: %3.5f" % (curRun, cost, costRatio, cvError, errRatio)
            
                
        # truncate the cost array
        self.costOut = self.costOut[:curRun]                        
        
        # Return list of training results
        return list([curRun, cost, prior_cost, xRows, yRows, bBreakEarly, xTrain, yTrain])        
        
    """
    Train the model using Stochastic Gradient Descent
    """        
    def trainModelSGD2(self, x, y, maxRuns=100, minError=None,
                   alpha=0.001, regStrength=0.001, batchRatio=0.1,
                   printRun=False):
        
        # create minibatch training set
        now = dt.datetime.now()
        rand_state = int(round((now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds(),0))
        xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(
        x, y, test_size=1-batchRatio, random_state=rand_state)
        
        yTrain = np.asmatrix(yTrain).reshape(len(yTrain),1)
        yTest = np.asmatrix(yTest).reshape(len(yTest),1)

        # assign for later plotting
        self.x = xTrain
        self.y = yTrain
        
        # get training size
        xRows = np.shape(xTrain)[0]
        yRows = np.shape(yTrain)[0]

        # show size of set        
        print "Train size: %d" % len(yTrain)
        print "yTrain: %f" % np.sum(yTrain)
        print "Counts: Train:%d, Test:%d" % (len(yTrain), len(yTest))
        
        cost = 0
        bBreakEarly = False
        self.costOut = np.zeros(maxRuns)
        prior_cost = None
        costRatio = 1
        errRatio = 1
        for curRun in range(maxRuns):        
            # Forward Propigate
            self.yh = self.nnModel.forwardPropigation(xTrain, yTrain)
            
            # Get Cost
            cost = self.nnModel.getCost(yTrain, regStrength)
            self.costOut[curRun] = cost            

            # update cost and error ratio
            if (prior_cost is not None):
                costRatio = (cost - prior_cost)/prior_cost

            # calc cost ratio momentum
            prior_cost = cost                

            # Print large runs every 1000 
            if (printRun == False) and (curRun % 1000 == 0):
                 print "iteration %d: loss %f" % (curRun, cost)    

            # Backpropigate
            # print "Shape: y:%s, yh:%s" % (np.shape(self.y), np.shape(self.yh))
            self.nnModel.backPropigation(yTrain, alpha, regStrength)
            if (printRun):
                print "Run: %s, Cost: %3.5f, CostRatio: %2.5f, ErrorRatio: %3.5f" % (curRun, cost, costRatio, errRatio)
            
                
        # truncate the cost array
        self.costOut = self.costOut[:curRun]                        
        
        # Return list of training results
        return list([curRun, cost, prior_cost, xRows, yRows, bBreakEarly, xTrain, yTrain])        
        
    def resplitData(self, cvRatio):
        """
        Resplit the initial data set to avoid a local minima
        """
        xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(
            self.x_copy, self.y_copy, test_size=cvRatio/2, random_state=None)

        yTrain = np.asmatrix(yTrain).reshape(len(yTrain),1)
        yTest = np.asmatrix(yTest).reshape(len(yTest),1)
        
        # assign for later plotting
        self.x = xTrain
        self.y = yTrain
        
        return xTrain, xTest, yTrain, yTest
        
    def plotTraining(self, useLog=False):
        # plot outputs
        maxConnections = len(self.nnModel.Connections)
        subPlots = maxConnections + 3
        plotNum  = 1
        
        plt.figure(figsize=(10, 15), dpi=100)
        
        # plot cost function
        plt.subplot(subPlots, 1, plotNum)
        if (useLog):
            plt.plot(np.log(self.costOut), 'r-')
        else:
            plt.plot(self.costOut, 'r-')
            
        # increment plot number for next subplot            
        plotNum += 1
        
        colors = ['g-','b-','k-','y-']
        
        for c in range(maxConnections):
            con = self.nnModel.Connections[c]
            plt.subplot(subPlots, 1, plotNum+c)
            if (useLog):
                plt.plot(np.log(con.g_WRatio), colors[c])
            else:
                plt.plot(con.g_WRatio, colors[c])
        
        plotNum += 1
        plt.subplot(subPlots, 1, plotNum+c)
        plt.plot(self.y, self.yh, 'y.')
        # plot the weight gradient ratio    
        #c1 = self.nnModel.Connections[0]
        #c2 = self.nnModel.Connections[1]
        #plt.subplot(subPlots, 1, maxConnections+2)    
        #plt.plot(np.divide(c1.g_WRatio, c2.g_WRatio))