#!/usr/bin/env python
# ## - Import Libraries

import numpy as np
import pandas as pd
import math
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix  

# ## - Read File (without header) given a delimeter
# 
# Reads a file with a given delimeter and returns the converted numpy array.
# Default when no delimeter is given it reads file as in csv format.

def readFile(trainFile,seperator=",",sample=False):
    try:
        data=pd.read_csv(trainFile, sep=seperator, header=None)
        if sample:
            data=data.sample(frac=1)
        return data.values
    except:
        print("Error reading training data file")


# ## - Distance Measure
# 
# Distance measures we have used is general **Minkowski distance** to calculate the distance between 2 set of records.
# <br>
# User can provide various values of p to use the desired form of minkowski as follows:
# 
# * **Manhattan:** Provide p=1.
# 
# * **Euclidean:(default)** Provide p=2.
# 
# * **ChebyShev:** Provide p=3.


def distMeasure(dataset,testdataset,targetIndex,p):
    n=len(testdataset)
    dist=0
    for i in range(n):
        if i==targetIndex:continue
        dist+=math.pow(abs(dataset[i]-testdataset[i]),p);
    return math.pow(dist,1/p)



def chebyshev(dataset,testdataset,targetIndex,p):
    n=len(testdataset)
    dist=0
    for i in range(n):
        if i==targetIndex:continue
        dist=max(abs(dataset[i]-testdataset[i]),dist);
    return dist


# ## -Train Test split
# 
# Given a percentage, it will split out the data into train dataset and test dataset.


def splitTrainTest(data,percent):
    total=len(data)
    trainTotal=int(total*percent*0.01)
    testTotal=total-trainTotal
    return (data[0:trainTotal],data[trainTotal:total])


# ## -K nearest neighbour Algorithm
# 
# In our algorithm we determine the k train data records which are nearest to the given test record.<br>
# We can use various distance measures by varying the value of p as we discussed above. (Default is 2 for Euclidean).
# <br>
# We have stored the distances of all the train data records from test record in a dictionary whose key is a tuple:
#  
# $$ [ (trainRecord1,0/1):distance1 ] $$
# 
# $$ [ (trainRecord2,0/1):distance2 ] $$
# 
# ..and so on
# 
# Then we will sort this dictionary in non-decreasing order of distances (values).
# And finally we pick the top **' k '** key, value pair of the dictionary and return the class/label appeared most number of times.  


def KNNAlgo(train,testRecord,k,targetIndex,p):
    dists={}
    count=0
    if p==3:
        funct=chebyshev
    else:
        funct=distMeasure
    for trainRecord in train:
        dist=funct(trainRecord,testRecord,targetIndex,p)
        dists[(str(trainRecord),trainRecord[targetIndex],count)]=dist
        count+=1
    sortedDict= sorted(dists.items(), key=operator.itemgetter(1))
    labelDict={}
    for i in range(k):
        if sortedDict[i][0][1] in labelDict.keys():
            labelDict[sortedDict[i][0][1]]+=1
        else:
            labelDict[sortedDict[i][0][1]]=1
    return max(labelDict.items(),key=operator.itemgetter(1))[0]


# ### -Prediction
# 
# This method will predict the label of each of the test record in test Dataset there by calling above KNN algorithm and returns a tuple of ``` accuracy, precision, recall and f1 Score ```.


def predict(train,test,k,targetIndex,p,dataType="robot"):
    count=0
    TP=0
    TN=0
    FP=0
    FN=0
    totalP=0
    totalN=0
    precision=None
    recall=None
    f1Val=None
    if dataType=="robot":
        for testRecord in test:
            predicted=KNNAlgo(train,testRecord,k,targetIndex,p)
            actual=testRecord[targetIndex]
            if actual==0:
                totalP+=1
            elif actual==1:
                totalN+=1
            if actual==predicted:
                count+=1
                if predicted==0:
                    TP+=1
                else:
                    TN+=1
        FP=totalN-TN
        FN=totalP-TP
        if TP+FP!=0:
            precision=TP/(TP+FP)
        if TP+FN!=0:
            recall=TP/(TP+FN)
        if recall and precision:
            f1Val=2*recall*precision/(recall+precision)
        accuracy=count/len(test)
        return (accuracy,precision,recall,f1Val)
    #iris
    else:
        ta=0
        tb=0
        tc=0
        fa=0
        fb=0
        fc=0
        precisionA=None
        precisionA=None
        precisionA=None
        recallA=None
        recallB=None
        recallC=None
        f1ValA=None
        f1ValB=None
        f1ValC=None
        for testRecord in test:
            predicted=KNNAlgo(train,testRecord,k,targetIndex,p).upper()
            actual=testRecord[targetIndex].upper()
            # a=Iris-setosa
            # b=Iris-virginica
            # c=Iris-versicolor
            if actual==predicted:
                count+=1
                if actual=="IRIS-SETOSA":
                    ta+=1
                elif actual=="IRIS-VIRGINICA":
                    tb+=1
                elif actual=="IRIS-VERSICOLOR":
                    tc+=1
            else:
                if predicted=="IRIS-SETOSA":
                    fa+=1
                elif predicted=="IRIS-VIRGINICA":
                    fb+=1
                elif predicted=="IRIS-VERSICOLOR":
                    fc+=1
        
        if ta+fa!=0:
            precisionA=ta/(ta+fa)
        if tb+fb!=0:
            precisionB=tb/(tb+fb)
        if tc+fc!=0:
            precisionC=tc/(tc+fc)

        if ta+fb+fc!=0:
            recallA=ta/(ta+fb+fc)
        if tb+fa+fc!=0:
            recallB=tb/(tb+fa+fc)
        if tc+fa+fb!=0:
            recallC=tc/(tc+fa+fb)

        if recallA and precisionA:
            f1ValA=2*recallA*precisionA/(recallA+precisionA)
        if recallB and precisionB:
            f1ValB=2*recallB*precisionB/(recallB+precisionB)
        if recallC and precisionC:
            f1ValC=2*recallC*precisionC/(recallC+precisionC)
#         print(ta,fb,fc)        
#         print(fa,tb,fc)        
#         print(fa,fb,tc)        
        head=("TYPE","Precision","Recall","F1-score")
        aM=("IRIS-SETOSA",precisionA,recallA,f1ValA)
        bM=("IRIS-VIRGINICA",precisionB,recallB,f1ValB)
        cM=("IRIS-VERSICOLOR",precisionC,recallC,f1ValC)
        
        return ((count/len(test)),head,aM,bM,cM)


# ### - Train And Predict
# 
# Interface for both the kind of datasets i.e **Robot and iris.**<br>
# It will clean up the datasets by removing unnecessary columns. <br>
# And finally call the above methods with appropriate parameters (According to the user call).


def trainAndPredict(trainFile,percent,k,targetIndex,dataType,p=2,testFile=None):
    testData=[]
    if dataType=="robot":
        data=readFile(trainFile," ")
        data=np.delete(data, 0, 1)
        data=np.delete(data, 7, 1)
        targetIndex=0
        if testFile:
            testData=readFile(testFile," ")
            testData=np.delete(testData, 0, 1)
            testData=np.delete(testData, 7, 1)
            
    elif dataType=="iris":
        data=readFile(trainFile)
        if testFile:
            testData=readFile(testFile)
    else:
        print("Invalid data type : expected iris or robot")
        return
    
    train,validate=splitTrainTest(data,percent)
    if len(testData)>0:
        validate=testData
    if len(validate)==0:
        print("No validate/test data to evaluate")
        return None
    return predict(train,validate,k,targetIndex,p,dataType)

def skLearn(trainFile,trainDataPercent,delimeter,unwantedCols,NANCols,k,targetIndex,avg=None,testFile=None):
    if trainDataPercent==100:
        print("No Validation data (Training percent should be less than 100)")
        return
    data=readFile(trainFile,delimeter)
#     data=data.sample(frac=1)
        
    le = LabelEncoder()
    X=None
    
    for i in NANCols:
        data[:,i]=le.fit_transform(data[:,i])
    
    Y=data[:,targetIndex]
    Y=Y.astype('int')
    X=np.delete(data,unwantedCols,axis=1)
    trainX,testX,trainY,testY=train_test_split(X,Y,test_size=(100-trainDataPercent)/100,shuffle=False)
    if testFile:
        testData=readFile(testFile,delimeter)
        for i in NANCols:
            testData[:,i]=le.fit_transform(testData[:,i])
        testY=testData[:,targetIndex]
        testY=testY.astype('int')
        testX=np.delete(testData,unwantedCols,axis=1)
        
    
    #specify p=1,2,3
    tree = KNeighborsClassifier(n_neighbors=k,p=2)      
    tree.fit(trainX, trainY)  
    y_pred = tree.predict(testX)  
    matrix=precision_recall_fscore_support(testY,y_pred,average=avg)
    confMat=confusion_matrix(testY,y_pred)
    print(confMat)
    print("=======================================================")
#     print(confMat)
    print("Accuracy= ",accuracy_score(testY,y_pred))
    print(classification_report(testY, y_pred))

# ***
# 
# ## Part-1 Working with Robot-1, Robot-2 and Iris Datasets.
# 
# Please run the below cell by passing following parameters.
# 
# 1. Train File: Path of train File
# 2. Percentage: use for training **(remaining percentage will be used for the validation if test file is not provided).**
# 3. Value of K
# 4. Target Index (must not change)
# 5. Identifier: "robot" or "iris" to distinguish robots and iris dataset.
# 6. P: distance measure 1,2 (default),3... and so on.
# 7. Test File (optional): Path of test File.

# ### 1.1 Robot-1


#uncomment below to run it on a test file passed as last parameter
print("Robot1")
testFile=None
if len(sys.argv)>1:
    print("Test data evaluation")
    testFile=sys.argv[1]
else:
    print("Validation data evaluation")
matrix=trainAndPredict("RobotDataset/Robot1",80,7,1,"robot",2,testFile)
if matrix:
    print("=================================")
    print("Accuracy= ",matrix[0])
    print("Precision= ",matrix[1])
    print("Recall= ",matrix[2])
    print("F1-Score= ",matrix[3])

print("=======================================================")
print("Scikit Learn for Robot1")
print("=======================================================")
print("Accuracy and measurement matrix for Robot1")
skLearn("RobotDataset/Robot1",80," ",[0,1,8],[],7,1,"binary",testFile)
print("=======================================================")
