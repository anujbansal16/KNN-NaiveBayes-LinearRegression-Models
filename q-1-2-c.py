#!/usr/bin/env python
# ## - Import Libraries

import numpy as np
import pandas as pd
import math
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


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


mpl.rcParams.update(mpl.rcParamsDefault)
def drawGraph(trainFile,percent,targetIndex,dataType,title=None,testFile=None):
    accuracyList1=[]
    accuracyList2=[]
    accuracyList3=[]
    if dataType=="robot":
        for i in range(1,18,2):
            matrix1=trainAndPredict(trainFile,percent,i,targetIndex,"robot",2,testFile)
            matrix2=trainAndPredict(trainFile,percent,i,targetIndex,"robot",1,testFile)
            matrix3=trainAndPredict(trainFile,percent,i,targetIndex,"robot",3,testFile)
            accuracyList1.append(matrix1[0])
            accuracyList2.append(matrix2[0])
            accuracyList3.append(matrix3[0])
    elif dataType=="iris":
        for i in range(1,18,2):
            matrix1=trainAndPredict(trainFile,percent,i,targetIndex,"iris",2,testFile)
            matrix2=trainAndPredict(trainFile,percent,i,targetIndex,"iris",1,testFile)
            matrix3=trainAndPredict(trainFile,percent,i,targetIndex,"iris",3,testFile)
            accuracyList1.append(matrix1[0])
            accuracyList2.append(matrix2[0])
            accuracyList3.append(matrix3[0])
    else:
        print("Invalid data type : expected iris or robot")
        return
    plt.title(title)
    plt.xlabel("number of neighbours (K)")
    plt.ylabel("Accuracy (in percent)")
    plt.plot(range(1,18,2), accuracyList1,color="orange", linewidth=2.5, label="Euclidean")
    plt.plot(range(1,18,2), accuracyList2,color="green", linewidth=2.5, label="Manhattan")
    plt.plot(range(1,18,2), accuracyList3,color="blue", linewidth=2.5, label="ChebyShev")
    plt.legend()
    plt.grid(True)
    plt.show()
    

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
print("Iris")
testFile=None
if len(sys.argv)>1:
    print("Test data evaluation visualization")
    testFile=sys.argv[1]
else:
    print("Validation data evaluation visualization")
drawGraph("Iris/Iris.csv",80,4,"iris","IRIS",testFile)