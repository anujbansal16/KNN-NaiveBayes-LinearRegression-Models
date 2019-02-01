#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

# ## Part-3 Regression

def splitTrainTest(data,percent):
    total=len(data)
    trainTotal=int(total*percent*0.01)
    testTotal=total-trainTotal
    return (data[0:trainTotal],data[trainTotal:total])


# function return Mean absolute Value
def MAE(testRecords,testYs,predictions):
    error=0
    for actual,predicted in zip(testYs,predictions):
        predicted=predicted[0]
        error+=abs(actual-predicted)
    print("Mean Absolute Error = ",error/len(testYs))



# function return Mean square error
def MSE(testRecords,testYs,predictions):
    error=0
    for actual,predicted in zip(testYs,predictions):
        predicted=predicted[0]
        error+=(actual-predicted)**2
    print("Mean Square Error = ",error/len(testYs))



# function return Mean percentage error
def MPE(testRecords,testYs,predictions):
    error=0
    for actual,predicted in zip(testYs,predictions):
        predicted=predicted[0]
        error+=(actual-predicted)/actual
    print("Mean Percentage Error = ",100*error/len(testYs))


# ### 1. Predicting probabiliy of getting admit


import copy
def predictProbAdmit(trainFile,percent,independentVariable=[1,2,3,4,5,6,7],targetIndex=8,forGraph=False,testFile=None):
    data=pd.read_csv(trainFile).values
    independentVariable=[0]+independentVariable
    
    train,test=splitTrainTest(data,percent)
    
    if testFile:
        test=pd.read_csv(testFile).values
    
    otest=copy.deepcopy(test)
    otrain=copy.deepcopy(train)
    
    testY=test[:,targetIndex]
    trainY=train[:,targetIndex]
    
    test[:,0]=1 #changeing first column to constant so it can be used for intercept
    train[:,0]=1
    
    train=train[:,independentVariable]
    test=test[:,independentVariable]
   
    if forGraph:
        test=train
        testY=trainY
        otest=otrain
    
    y=np.transpose(np.matrix(trainY))
    X=np.matrix(train)
    XT=np.transpose(X)
    inverse=np.linalg.inv(XT*X)
    coefficents=inverse*XT*y
    predicted=np.array(np.matrix(test)*coefficents)
    coefficents=np.array(coefficents)
    return (otest,testY,predicted,coefficents)#     



def printActualPredicted(testRecords,testYs,predictions,coefficents):
    print("Intercept value= ",coefficents[0][0])
    print("GRE Score Coefficent ",coefficents[1][0])
    print("TOEFL Score Coefficent ",coefficents[2][0])
    print("University Rating Coefficent ",coefficents[3][0])
    print("SOP Coefficent ",coefficents[4][0])
    print("LOR Coefficent ",coefficents[5][0])
    print("CGPA Coefficent ",coefficents[6][0])
    print("Research Coefficent ",coefficents[7][0])
    print("=======================================================================================")
    MSE(testRecords,testYs,predictions)
    print("=======================================================================================")
    print("%20s | %22s | %10s"%("Admit Chance(Actual)","Admit Chance(Predicted)","Test Records"))
    print("=======================================================================================")
    for testRecord,actual,pre in zip(testRecords,testYs,predictions):
        print("%20f | %22f | %10s"%(actual,pre[0],testRecord))



# ### 3.Behaviour of the coefficents:


def visualize(trainFile,percent,onTraining=True):
    labels={1:"GRE Score",2:"TOEFL Score",3:"University Rating",4:"SOP",5:"LOR",6:"CGPA",7:"Research"}
    
    testRecords,testYs,predictions=predictProbAdmit(trainFile,percent,[1,2,3,4,5,6,7],8,onTraining)[:3]
    plt.scatter(testRecords[:,1],testYs,color="orange",label="Actual")
    plt.scatter(testRecords[:,1],predictions,color="green",label="Predicted")
    if onTraining:
        plt.xlabel("Training dataset")
    else:
        plt.xlabel("Validation dataset")
    plt.ylabel("Chance of Admit")
    plt.legend()
    plt.show()    
    
    for i in range(1,3):
        testRecords,testYs,predictions=predictProbAdmit(trainFile,percent,[i],8,onTraining)[:3]
        plt.scatter(testRecords[:,i],testYs,color="orange")
        plt.plot(testRecords[:,i],predictions,color="green",linewidth="2.4",label="Predicted")

        plt.xlabel(labels[i])
        plt.ylabel("Chance of Admit")
        plt.legend()
        plt.grid(True)

        plt.show()



visualize("AdmissionDataset/data.csv",100,True)