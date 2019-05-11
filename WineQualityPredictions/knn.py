# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:13:29 2018

@author: owner
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:00:09 2018

@author: owner
"""

import csv
import math
import collections as coll
import random
from datetime import datetime

startTime = datetime.now()

def minMaxProcess(file):
    
    # finds the mean and STDev for the data set
    
    wineCat = dict()
    maxMin = dict()
    
    with open(file) as csvfile:
      reader = csv.DictReader(csvfile, delimiter = ';')
      for row in reader:
        for key in row:
            wineCat.setdefault(key, []).append(float(row[key]))
    
    # calculates the Mean of each category in the data set
    
    for key in wineCat:
        wineCat[key].sort()
    
    for key in wineCat:
        maxMin.setdefault(key, []).append(wineCat[key][0])
        maxMin.setdefault(key, []).append(wineCat[key][-1])
            
    return(maxMin)


def normProcess(wine, minMax):
    
    # normalizes all of the wine data based on the mean/STDev of that particilar
    # category in an attempt to remove bias from the data set.
    
    for key in wine:
        if key != 'quality':
            a = wine[key]
            min1 = minMax[key][0]
            max1 = minMax[key][1]
            # add STDeviation
            
            if (a-min1) == 0:
                wine[key] = .0001
            else:
                wine[key] = round((a - min1)/(max1 - min1),3)
    
    return(wine)


def floatProcess(wine):
    
    #changes all wine data from strings into float with the exception
    
    for key in wine:
        wine[key] = float(wine[key])
    
    return(wine)


def kfoldData(file, minMax):
    
    # Partitions data into k = 4 folds. Three of the partitions will
    # be used for Training and one partition will be used for Testing.
    
    winelist1 = []
    winelist2 = []
    winelist3 = []
    winelist4 = []
       
    count = 0
    
    with open(file) as csvfile:
      reader = csv.DictReader(csvfile, delimiter = ';')
      for row in reader:
        
        row = floatProcess(row)
        row = normProcess(row, minMax)
        
        if count < 4898/4:
            winelist1.append(row)
        elif count >= 4898/4 and count < 2*(4898/4):
            winelist2.append(row)
        elif count >= 2*(4898/4) and count < 3*(4898/4):
            winelist3.append(row)
        else: 
            winelist4.append(row)
            
        count = count + 1
    
    return(winelist1, winelist2, winelist3, winelist4)


def dataSplit(training):
    
    # "randomly" splits ~20% of the training set off into a validation set
    
    valSet = []
    trSet = []
    length = len(training)
    for x in range(0, length):
        r =  random.randint(0,1)
        
        if r == 0 and len(valSet) <= length/4:
            valSet.append(training[x])
        else:
            trSet.append(training[x])
    
    return(valSet, trSet)
        
        
def eucDistance(wine1, wine2):

    # calculates the euclidean distance between two wines.
    
    d = 0
    
    for key in wine1:
        if key != 'quality':
            a = wine1[key]
            b = wine2[key]
            d += (a - b)**2
            
    return(math.sqrt(d))

def cosDistance(wine1, wine2):
    
    # calculates the manhatten distance between two wines

    d = 0
    topSum = 0
    botSum1 = 0
    botSum2 = 0
    for key in wine1:
        if key != 'quality':
            a = wine1[key]
            b = wine2[key]
            topSum += (a*b)
            botSum1 +=  a**2
            botSum2 += b**2
    
    d = topSum/(math.sqrt(botSum1)*math.sqrt(botSum2))
     
    return(d)    
 
def manDistance(wine1, wine2):
    
    # calculates the manhatten distance between two wines

    d = 0
    
    for key in wine1:
        if key != 'quality':
            a = wine1[key]
            b = wine2[key]       
            d += abs((a - b))
            
    return(d)       
    

def nearNb(wine,trSet, k, dCalc):
    
    # finds the k nearest neighbors and returns the index of where they are
    # in the provided training set.
    
    d = []
    nN = []
    for x in range(len(trSet)):
        if dCalc == 0:
            d.append(cosDistance(wine,trSet[x]))
        elif dCalc == 1:
            d.append(eucDistance(wine,trSet[x]))
        else:
            d.append(manDistance(wine,trSet[x]))
    
    for n in range(0,k):
        
        nN.append(min(xrange(len(d)), key=d.__getitem__))
        d[nN[n]] = float('inf')
    
    return(nN)


def commClass(nN,trSet):
    
    # takes a vote of the k nearest neighbors and returns the highest voted
    # quality classifier.
    
    count = coll.Counter()
    for nearest in nN:
        vote = trSet[nearest]['quality']
        count[vote] += 1
    
    return(count.most_common(1)[0][0])


def kNN(valSet, trSet, k, dCalc):
    
    # Takes the validation set from the training set and runs KNN on each
    # data point in the validation set. Returns a list of the KNN 
    # classifications for all validation data points.
    
    nClass = []
    for entry in valSet:
        neighbors = nearNb(entry,trSet,k, dCalc)
        nClass.append(commClass(neighbors, trSet))
    
    return(nClass)
 
    
        
def knnAccuracy(valSet, valSetClassified):
    tTotal = 0.0
    fTotal = 0.0
    trueList = []
    fList = []
    
    for entry in range(0,len(valSet)):
        trueList.append(valSet[entry]['quality'])
        
    for entry in range(0,len(valSet)):
        if trueList[entry] == valSetClassified[entry]:
            tTotal = tTotal + 1
    
    for qLabels in set(trueList): # quality only ranges from 3 to 9 
        
        f1Score = f1IndScore(qLabels, trueList, valSetClassified)        
        fList.append(f1Score)

    fTotal = sum(fList)
    
    accuracy = tTotal/(len(trueList)) 
    f1ScoreAve = float(fTotal)/(len(set(trueList)))    
    return(100*round(accuracy,2), 100*round(f1ScoreAve,2))

def f1IndScore(quality, trueList, valSetClassified):
    
    T,F = 0.0,0.0
    
    for entry in range(0,len(trueList)):
        if trueList[entry] == quality:
            if valSetClassified[entry] == trueList[entry]:
                T = T + 1   
            else:
                F = F + 1
                
        if valSetClassified[entry] == quality:
            if valSetClassified[entry] != trueList[entry]:
                F = F + 1
    f1 = 2*T/((2*T)+F)
    return(f1)          

def dataPerm(s1, s2, s3, s4):
    
    training = []
    test = []
    
    training.append(s2+s3+s4) 
    test.append(s1)
    
    training.append(s1+s3+s4) 
    test.append(s2)
    
    training.append(s1+s2+s4) 
    test.append(s3)

    training.append(s1+s2+s3) 
    test.append(s4)
    
    return(training, test)

def dataOut():
    
    file = 'winequality-white.csv'
    minMax = minMaxProcess(file)
    s1,s2,s3,s4 = kfoldData(file, minMax)
    training, test = dataPerm(s1, s2, s3, s4)
    
    return(training, test)

def nicePrint(chosenK, dCalc, F_scoreV, AccuracyV, F_scoreTE, AccuracyTE):
    
    AAverageV = sum(AccuracyV)/len(AccuracyV)
    AAverageTE = sum(AccuracyTE)/len(AccuracyTE)
    FAverageV = sum(F_scoreV)/len(F_scoreV)
    FAverageTE = sum(F_scoreTE)/len(F_scoreTE)
    
    
    
    if dCalc == 0:
        chosenDM = 'Cosine Similarity'
    elif dCalc == 1:
        chosenDM = 'Euclidean Distance'
    else:
        chosenDM = 'Manhatten Distance'
        
    print('Hyper-parameters:')
    print('K:' + ' ' + str(chosenK))
    print('Distance Measure:' + ' ' + chosenDM)
    print('')
    
    for fold in range(0,len(F_scoreV)):
        foldName = fold + 1
        print('Fold' + '-' + str(foldName))
        print('Validation: ' + '' + 'F1 Score: ' + str(F_scoreV[fold]) + ','
              + ' Accuracy: ' + str(AccuracyV[fold]))
        
        print('Test: ' + '' + 'F1 Score: ' + str(F_scoreTE[fold]) + ','
              + ' Accuracy: ' + str(AccuracyTE[fold]))
        print('')

    print('Average')
    print('Validation: ' + '' + 'F1 Score: ' + str(FAverageV) + ','
          + ' Accuracy: ' + str(AAverageV))
    
    print('Testing: ' + '' + 'F1 Score: ' + str(FAverageTE) + ','
          + ' Accuracy: ' + str(AAverageTE))
    print('')      

      
        
def run():
    
    training, test = dataOut()
    
    k = [1]             # adjust K array to Tune
    dCalc = [2]         # adjust dCalc to change distance Calc Method. #
                        # Manhatten Distance best for all folds
                        
    AccuracyV = []      # Validation Set Accuracy
    F_scoreV = []       # Validation F1Score
    AccuracyTE = []     # Test Set Accuracy
    F_scoreTE = []      # Test Set F1Score

    for tset in range(0,4):
    
        valSet, trSet = dataSplit(training[tset])
        
        valSetClassified = kNN(valSet,trSet,k[0], dCalc[0])
        A, F = knnAccuracy(valSet, valSetClassified)
        AccuracyV.append(A)
        F_scoreV.append(F)
        
        testSetClassified = kNN(test[tset],trSet,k[0], dCalc[0])
        A, F = knnAccuracy(test[tset], testSetClassified)
        AccuracyTE.append(A)
        F_scoreTE.append(F)
        
    
    return(F_scoreV, AccuracyV, F_scoreTE, AccuracyTE)
    
 # training0 = 3,2
 # training1 = 3,2
 # training2 = 3,2
 # training3 = 3,2    

              
def main():
    
    F_scoreV, AccuracyV, F_scoreTE, AccuracyTE = run()
    nicePrint(1,2,F_scoreV, AccuracyV, F_scoreTE, AccuracyTE)

main()

print(datetime.now()-startTime)

#cp.run('main()')

#plt.plot(neighbors,Accuracy)
#plt.plot(neighbors,F_score)

