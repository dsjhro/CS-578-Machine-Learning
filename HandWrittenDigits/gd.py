# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:38:58 2018

@author: owner
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:02:56 2018

@author: owner
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:04:09 2018

@author: owner
"""

import struct
import gzip
import numpy as np
import os
import sys
import random as rand
import math 
from matplotlib import pyplot as plt




def readImage(filename):
    
    '''
    Funtion reads MNIST Data and parses it into a 2D Numpy Array
    Courtesy of Github and Tyler Neylon - Modified to Unzip gz and updated
    legacy function 'fromString' to 'fromBuffer' for stability. 
    
    '''

    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
def dataProcess(data, featureType, bias):
    
    '''
    Processes 2D arrays that represent the MNSIT images and processes them 
    into a numpy array filled with flattened 1D arrays with 784 elements in it.
    It then takes the image data and divides by 255 to normalize all of the 
    arrays.  
    '''

    if featureType == 'type1':
        
        # testData Temp List
        tempData = []
        
        # Takes 2D numpy arrays and flattens into list of 1D array with 784 
        # Elements in it. 
        for instance in range(0, 10000):
            tempData.append(data[instance].flatten())
        
        # Changes List of 1D numpy arrays into Numpy Array and divides by 255
        tempData = np.array(tempData)
        tempData = tempData/255.0
    
    if featureType == 'type2':
        
        # testData Temp List
        tempData = []
        
        for instance in data:       
            
            # Takes shape of Data
            i, j = instance.shape
            
            # Defines Window Size
            K = 2
            L = 2
            
            # Partitions 2D data array into even number of windows
            iK = i // K
            jL = j // L
            
            # Reshapes 2D array into smaller windowed array based on max of window
            b = instance[:iK*K, :jL*L].reshape(iK, K, jL, L).max(axis=(1, 3))
            
            # appends flattened reshaped array into list of of instances
            tempData.append(b.flatten())
        
        # Changes List of 1D numpy arrays into Numpy Array and divides by 255
        tempData = np.array(tempData)
        tempData = tempData/255.0
        
    if bias == 1:
        
        # Appends Bias value of 1 to end of 1D array to create a 785 Element Array        
        tD = []
        for entry in range(0,len(tempData)):
            temp = np.append(tempData[entry],1)
            tD.append(temp)
        
        tempData = tD
     
    procData = np.array(tempData)
        
    return(procData)
    
def dataOut(DATA_FOLDER, featureType, bias):
    
    # Changes the Directory to the directory where the Data is located
    os.chdir(DATA_FOLDER)

    # Read in Image Data for Training and Testing
    trainData = readImage('train-images-idx3-ubyte.gz')
    trainLabels = readImage('train-labels-idx1-ubyte.gz')
    testData = readImage('t10k-images-idx3-ubyte.gz')
    testLabels = readImage('t10k-labels-idx1-ubyte.gz')
    
    # Process Crude Data into flattened BITMAPS that are in the correct form 
    # and size for training 
    trainLabels = trainLabels[0:10000]    
    procTrainData = dataProcess(trainData, featureType, bias)
    procTrainData = procTrainData[0:10000]
    
    # Process Crude Data into flattened BITMAPS that are in the correct form 
    # and size for Testing. 
    procTestData = dataProcess(testData, featureType, bias)      
    
    return(procTrainData, trainLabels, procTestData, testLabels)
    
def shuffle(procTrainData, trainLabels, trainingSize):
    
    combine = list(zip(procTrainData,trainLabels))
    rand.shuffle(combine)
    procTrainData,trainLabels = zip(*combine)

    return(procTrainData[0:trainingSize],trainLabels[0:trainingSize])

def initWeight(featureType):
    
    '''
    Initializes the weight vector for digits 0 - 9. 
    The weight vector consists of 785 elements that are randomized
    between .25 and .95
    '''
    
    # init weight list
    weight = []
    
    if featureType == 'type1':
        length = 785
    else:
        length = 197
    
    # randomizes 785 elements in a weight array for each digit (0-9)
    for digit in range(0,10):
        weight.append(np.random.uniform(low=0.25, high=.95, size=(length,)))
    
    # transforms list into numpy array
    return(np.array(weight))
    
def initZeroWeight(featureType):
    
    '''
    Initializes the weight vector for digits 0 - 9. 
    The weight vector consists of 785 elements that are randomized
    between .25 and .95
    '''
    
    # init weight list
    weight = []
    
    if featureType == 'type1':
        length = 785
    else:
        length = 197
    
    # randomizes 785 elements in a weight array for each digit (0-9)
    for digit in range(0,10):
        weight.append(np.zeros(length))
        
    # transforms list into numpy array
    return(np.array(weight))

#computes the sigmoid function for the weight(w) and test instance(x)
def sigFunc(x,w):
    
    z = np.dot(x,w)
    try:
        f = 1.0/(1.0 + math.exp(-z))
    except OverflowError:
        if z > 0:
            f = .99999999999
        else:
            f = .000000000001
            
    if f == 1.000:
        f = .9999999999999
    
    if f == 0.0000:
        f = .00000000000001
    
    return(f)

def gradient(x,w,y):
    
    try:
        dot = math.exp(-np.dot(x,w))
    except OverflowError:
        if np.dot(x,w) > 0:
            dot = 0.0000001
        else:
            dot = 10000000.0
    
    grad = x*(((dot)*math.pow(sigFunc(x,w),2))*(-y/(sigFunc(x,w)) + (1-y)/(1-sigFunc(x,w))))
    
    return(np.array(grad))


def errorLoss(x,w,y):
    
    err = -y*math.log(sigFunc(x,w)) - (1-y)*math.log(1 - sigFunc(x,w))

    return(err)


def digitGradient(instance, label, weight):
    
    '''
    Performs the perceptron algorithm on all 10 digits in the MNSIT Data Set
    Using one instance of the data.
    Returns the updated weights and the perceptron index (0 - 9) that had the
    highest total sum.
    '''
    
    gradSum = []
    y_gold = np.zeros(10)
    y_gold[label] = 1
        
    for digit in range(0,10):
        
        gradSum.append(gradient(instance, weight[digit], y_gold[digit]))
    
    return(np.array(gradSum))

def digitErrorLoss (instance, label, weight):
    
    '''
    Performs the perceptron algorithm on all 10 digits in the MNSIT Data Set
    Using one instance of the data.
    Returns the updated weights and the perceptron index (0 - 9) that had the
    highest total sum.
    '''
    errorSum = np.zeros(10)
    y_gold = np.zeros(10)
    y_gold[label] = 1
        
    for digit in range(0,10):
        
        errorSum[digit] = errorLoss(instance, weight[digit], y_gold[digit])
    
    return(errorSum)



def GradDesc(trainingData, trainingLabels, testingData, testingLabels, learningRate,featureType, regularization, lamb):
    
    # Initialize weights
    weight = initWeight(featureType) # initWeight
    
    #Keeps track of count of loops and array of epochs
    count = 0
    epoch = []
    
    # init Loss Function delta
    minLoss = 10.0
    
    # Array of average loss after each iteration
    aveLoss = []
    
    # Training and Testing Accuracy Arrays
    trainingAccuracy = []
    testAccuracy = []
    
    # Max Iterations for while loop 
    maxIter = 199

    while minLoss >.10  and count <= maxIter:
        
        
        # Initialize Delta 
        delta = initZeroWeight(featureType)
        
        # Total Loss
        totLoss = np.zeros(10)
        
        #trainingData, trainingLabels = shuffle(trainingData,trainingLabels,10000)
        
        #Calculate the Gradient 
        for instance in range(0,len(trainingData)):
            
            y_gold = np.zeros(10)
            y_gold[trainingLabels[instance]] = 1
            
            for digit in range(0,10):
                
                delta[digit] +=  gradient(trainingData[instance], weight[digit], y_gold[digit])
        
        delta = learningRate * (delta/float(len(trainingData)))
        
        #Regularize delta if regularization == true
        if regularization == True:
            L2 = lamb * weight
            
            #Remove Bias term from Regularizer
            L2[:, -1] = 0.0
            delta = delta + L2
        
        #Adjust the weight based on the new gradient
        weight = weight - delta
        
        #Calculate Total Loss
        for instance in range(0,len(trainingData)):
            
            y_gold = np.zeros(10)
            y_gold[trainingLabels[instance]] = 1
            
            for digit in range(0,10):
                
                totLoss[digit] += errorLoss(trainingData[instance], weight[digit], y_gold[digit])
        
        
        #Regularize Loss function if regularization == true
        if regularization == True:
            L2Loss = 0.0
            for digit in range(0,9):
                L2Loss += lamb/2.0 * np.sum(np.square(weight[digit]))
            
            L2Loss = L2Loss/10.0
            
        else:
            L2Loss = 0.0
                
        # Appends calculated Avergae Error Loss to average loss array
        aveLoss.append((np.mean(totLoss)+L2Loss)/len(trainingData))
        
        # Training Accuracy Calculation Every Epoch
        trainGuesses = genDescTest(trainingData, weight)
        testGuesses = genDescTest(testingData, weight)
        
        # Testing Calculation Every Epoch
        trainingAccuracy.append(Accuracy(trainingLabels, trainGuesses))
        testAccuracy.append(Accuracy(testingLabels, testGuesses))
        
        # stopping condition
        minLoss = aveLoss[count]
        
        # Keeps Track of Iterations of While Loop
        count = count + 1
        epoch.append(count)
        
        print('Epoch ' + str(count) + ':' ' Training Loss: ' + str(round(aveLoss[count-1],2)) + ' Training Accuracy: '  + str(round(trainingAccuracy[count-1],2)) + ' Testing Accuracy :' + str(round(testAccuracy[count-1],2))  )
    
    # Plot and Save PNG image for specific hyper parameters
    plt.plot(epoch,trainingAccuracy, label = 'Training' )
    plt.plot(epoch,testAccuracy, label = 'Testing' )
    plt.legend()
    
    if regularization == True:
        reg = 'With a Regularizer'
    else:
        reg = 'Without a Regularizer'
        
    if featureType == 'type1':
        typ = 'Type 1'
    else: 
        typ = 'Type 2'
        
    plt.title(str(typ) + ' Gradient Descent ' + reg)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim((0.0, 1.0))   # set the ylim to bottom, top
    plt.savefig('convergence.png')
    
    return(weight,aveLoss, epoch, trainingAccuracy, testAccuracy)
    
def genDescTest(data, weight):
    
    # Array of Guesses
    guesses = []
    
    # Loop THrough all training Examples
    for entry in range(0,len(data)):
        
        # Temp list of max dot product and its Digit
        maxList = []
        for digit in range(0,10):     
            
            # Evaluate dot product of training/test example against all weights
            z = np.dot(weight[digit],data[entry])
            
            #append dot product and digit to temp array
            maxList.append((z, digit))     
    
        # Guesses Label based on max dot product and appends to guesses array
        label = sorted(maxList, reverse = True)[0][1]
        guesses.append(label)
        
    return(guesses)

def Accuracy(labels, guesses):

    correct = 0.0
    
    for entry in range(0,len(labels)):
        
        if labels[entry] == guesses[entry]:
            correct = correct + 1.0
    
    accuracy = correct/float((len(labels)))
    return(round(accuracy,4))

if __name__ == "__main__":
        
    # Default input Data
    trainingSize = 10000
    learningRate = 2.5  #.001 or #.00015
    regularization = False
    featureType = 'type2'
    bias = 1
    lamb = .005            #.005
    path = os.getcwd()
    DATA_FOLDER = path + '/DATA_FOLDER'
    
    # Sets input function variables to the user input if all the required 
    # inputs are present. 
    if len(sys.argv) >= 4:
        regularization = bool(sys.argv[1])
        featureType = sys.argv[2]
        DATA_FOLDER = ''
        for entry in range(3,len(sys.argv)):
            DATA_FOLDER =  DATA_FOLDER + sys.argv[entry]
        
    # Changes the Directory to the directory where the Data is located
    os.chdir(DATA_FOLDER)
    
    # Reads Image Data, processes it and returns data that is ready for the
    # Machine Learning Algorithm
    trainingData, trainingLabels, testingData, testingLabels = dataOut(DATA_FOLDER, featureType, bias) 
    
    # Runs Gradient Descent
    weight, averageLoss, epoch, trainingAccuracy, testAccuracy = GradDesc(trainingData, trainingLabels, testingData, testingLabels, learningRate, featureType, regularization, lamb)
    
    # Return to Home Folder for further runs    
    os.chdir(path)