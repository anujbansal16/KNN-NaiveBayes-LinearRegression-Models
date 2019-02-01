#!/usr/bin/env python
# coding: utf-8

# ## Part 2- Naive Bayes classifier on bank dataset

# According to the Bayes theorem:
# 
# $$ P(Y \mid X_1,X_2,...X_n)= \frac{P(X_1,X_2,...X_n \mid Y)\, P(Y)}{P(X_1,X_2,...X_n)}$$
# 
# Under naive bayes assumption :
# 
# $$ P(X_1,X_2,...X_n \mid Y)= \prod_{i=1}^nP(X_i \mid Y)$$
# 
# We will calculate the above posterior under the hypothesis $Y = 1$ and $Y = 0$ and whichever is greater will become class for that evidance.
# 
# **Training**
# * **Categorical Features**
# <br>
# <hr>
#     -We will first seperate out the training records which have target Class values 0 and 1 , 
#     say zeroRecords and oneRecords.<br>
#     -Amongst all the zeroRecords we will find out unqiue value count for all catergorical features and put them into a dictionary with key 0.<br>
#     -Amongst all the oneRecords we will find out unqiue value count for all catergorical features and put them into a dictionary with key 1.<br>
#     -Now put this dictionary into $featureIndexDictionary$ with key as index of catergorical feature.<br>
#     -Now these dictionary can be easily used to calculate the likelyhood under hypothesis $Y = 1$ and $Y = 0$ for categorical features.
# 
# $$\{6:\{0: \{1:count01,2:count02,3:count03\},$$
# $$    1: \{1:count11,2:count12,3:count13\}$$
# $$   \}$$
# $$ \}$$
# 
# <hr>
# 
# * **Numerical Features**
# <br>
# <hr>
#     -We will first seperate out the training records which have target Class values 0 and 1 , 
#     say zeroRecords and oneRecords.<br>
#     -Amongst all the zeroRecords we will find out the mean and standard deviation for all the numerical features and put them into a dictionary with key 0.<br>
#     -Amongst all the oneRecords we will find out the mean and standard deviation for all the numerical features and put them into a dictionary with key 1.<br>
#     -Now put this dictionary into $meanStdDeviationIndex$ dictionary with key as index of numerical feature.<br>
#     -Now these dictionary can be easily used to calculate the likelyhood using gaussian normal distribution under hypothesis $Y = 1$ and $Y = 0$ for numerical features.
# 
# $$\{3:\{0: \{mean: ,std: \},$$
# $$    1: \{mean: ,std: \}$$
# $$   \}$$
# $$ \}$$
# 
# <hr>
# 
# 
# 
# 
# We will remove the id column from the data internally.
# <br>
# Now the user can specify the below indexes used for the attributes to specify which attributes are categorical and which are numerical.
# 
# **Indexing**
# * [0] Age**(default-numerical)**
# 
# * [1] Number of years of experience  **(default-numerical)**
# 
# * [2] Annual Income   **(default-numerical)**
# 
# * [3] ZIPCode
# 
# * [4] Family size  **(default-numerical)**
# 
# * [5] Avgerage spending per month  **(default-numerical)**
# 
# * [6] Education Level. 1: 12th; 2: Graduate; 3: Post Graduate  **(default-categorical)**
# 
# * [7] Mortgage Value of house if any  **(default-numerical)**
# 
# * [8] Did this customer accept the personal loan offered in the last campaign?  **Output label **
# 
# * [9] Does the customer have a securities account with the bank? **(default-categorical)**
# 
# * [10] Does the customer have a certificate of deposit (CD) account with the bank? **(default-categorical)**
# 
# * [11] Does the customer use internet banking facilities? **(default-categorical)**
# 
# * [12] Does the customer uses a credit card issued by UniversalBank? **(default-categorical)**
# 
import numpy as np
import pandas as pd
import math
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

def splitTrainTest(data,percent):
    total=len(data)
    trainTotal=int(total*percent*0.01)
    testTotal=total-trainTotal
    return (data[0:trainTotal],data[trainTotal:total])


# ### -Normal Distribution

def getProbabilityNormal(x, mean, std):
    exp = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
    return (1 / (math.sqrt(2*math.pi) * std)) * exp


# ### -Algorithm


import pprint
def bayesClassifier(trainFile,percent,categoricalList=[6,9,10,11,12],numericalList=[0,1,2,4,5,7],testFile=None):
    testData=[]
    #data=pd.read_csv(trainFile,",",header=0).sample(frac=1).values
    data=pd.read_csv(trainFile,",",header=0).values
    data=np.delete(data,0,axis=1)

    train,test=splitTrainTest(data,percent)
    
    ##Test file#############
    if testFile:
        #ignoring first row assuming wrong data, change to none if accurate
        testData=pd.read_csv(testFile,",",header=0).values
        testData=np.delete(testData,0,axis=1)
    if len(testData)>0:
        test=testData
    ##Test file#############

    zeroRecords=np.where(train[:,8]==0)
    oneRecords=np.where(train[:,8]==1)

    totalZeroRecords=len(zeroRecords[0])
    totalOneRecords=len(oneRecords[0])
    # print(totalOneRecords,totalZeroRecords)
    
    ############################################################################
    #Calculate all the possible probabilities of all the attributes for each hypothesis
    
    
    ############################## Categorical Attr start ##############################
    featureIndexDict={}

    # {3:{0: {1:count,2:,3:}
    #     1: {1:,2:,3:}
    #     }
    #}
    
    for i in categoricalList:
        CCardUser={}
        CCardUserONE=np.unique(train[oneRecords[0],i],return_counts=True)
        # print("CCardUserONE",CCardUserONE)
        CCardUser[1]={value:CCardUserONE[1][index] for index,value in enumerate(CCardUserONE[0])}

        CCardCUserZERO=np.unique(train[zeroRecords[0],i],return_counts=True)
        # print("CCardCUserZERO",CCardCUserZERO)
        CCardUser[0]={value:CCardCUserZERO[1][index] for index,value in enumerate(CCardCUserZERO[0])}
        featureIndexDict[i]=CCardUser
    # pprint.pprint(featureIndexDict)
    
    ############################## Categorical Attr end ##############################
    
    ############################## Numerical Attr start ##############################
    
    meanStdDictIndex={}
    
    # {0:{0: {"mean":,"std":}
    #     1: {"mean":,"std":}
    #     }
    # }
    
    for i in numericalList:
        myDict={}
        ageMeanONE=np.mean(train[oneRecords[0],i])
        # print("ageMeanONE ",ageMeanONE)
        ageMeanZERO=np.mean(train[zeroRecords[0],i])
        # print("ageMeanZERO ",ageMeanZERO)
        ageStdONE=np.std(train[oneRecords[0],i])
        # print("ageStdONE ",ageStdONE)
        ageStdZERO=np.std(train[zeroRecords[0],i])
        myDict[1]={"mean":ageMeanONE,"std":ageStdONE}
        myDict[0]={"mean":ageMeanZERO,"std":ageStdZERO}
        meanStdDictIndex[i]=myDict
#     pprint.pprint(meanStdDictIndex)
    ############################## Numerical Attr end ##############################
    ############################################################################
    
    priorOne=totalOneRecords/(totalZeroRecords+totalOneRecords)
    priorZero=totalZeroRecords/(totalZeroRecords+totalOneRecords)
    priors={0:priorZero,1:priorOne}
    totalOneZeroRecords={0:totalZeroRecords,1:totalOneRecords}

    ############################ Prediction ###################################
    count=0
    answers={}
    TP=0
    TN=0
    FP=0
    FN=0
    totalP=0
    totalN=0
    for testR in test:
        for i in [0,1]:
            numerical=1
            categorical=1
            for index in numericalList:
                numerical*=getProbabilityNormal(testR[index],meanStdDictIndex[index][i]["mean"],meanStdDictIndex[index][i]["std"])
            for index in categoricalList:
                try:
                    categorical*=featureIndexDict[index][i][testR[index]]/totalOneZeroRecords[i]
                except:
                    categorical=0
            ans=categorical*numerical*priors[i]
            answers[i]=ans
#         print(answers)
        actual=testR[8]
        if actual==0:
            totalN+=1
        if actual==1:
            totalP+=1
        if answers[1]>answers[0] and testR[8]==1:
            count+=1
            TP+=1
        elif answers[1]<=answers[0] and testR[8]==0:
            count+=1
            TN+=1
    FP=totalN-TN
    FN=totalP-TP
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1Val=2*recall*precision/(recall+precision)
    accuracy=count/len(test)
    return (accuracy,precision,recall,f1Val)
    
    ############################ Prediction end ###################################


print("Naive Bayes Classifier on LoanDataset")
testFile=None
if len(sys.argv)>1:
    print("Test data evaluation")
    testFile=sys.argv[1]
else:
    print("Validation data evaluation")
#make sure first row is proper in testData.csv
#with test data file as last parameter
matrix=bayesClassifier("LoanDataset/data.csv",80,[6,9,10,11,12],[0,1,2,4,5,7],testFile)

print("=================================")
print("Accuracy= ",matrix[0])
print("Precision= ",matrix[1])
print("Recall= ",matrix[2])
print("F1-Score= ",matrix[3])

