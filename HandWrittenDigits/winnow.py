# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 19:12:43 2018

@author: owner
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:47:42 2018

@author: owner
"""

import struct
import gzip
import numpy as np
import time
import os
import sys
#import matplotlib.pyplot as plt
import random as rand


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
    
def dataProcess(data):
    
    '''
    Processes 2D arrays that represent the MNSIT images and processes them 
    into a numpy array filled with flattened 1D arrays with 785 elements in it.
    784 features and a bias term. It then takes the image data and rounds them
    into 0's or 1's depending on their element value.  
    '''
    
    # testData Temp List
    tempData = []
    
    # Takes 2D numpy arrays and flattens into list of 1D array with 784 
    # Elements in it. 
    for instance in range(0, 10000):
        tempData.append(data[instance].flatten())
    
    # Changes List of 1D numpy arrays into Numpy Array. 
    tempData = np.array(tempData)
    
    # Rounds all values in 1D array into a 0 or 1 based on image data value  
    for instance in range(0,len(tempData)):
            tempData[instance][tempData[instance] <= 127] = 0    
            tempData[instance][tempData[instance] >= 128] = 1
    
    # Appends Bias value of 1 to end of 1D array to create a 785 Element Array        
    tD = []
    for entry in range(0,len(tempData)):
        temp = np.append(tempData[entry],1)
        tD.append(temp)
    
    # Changes final processed data into a numpy array
    procData = np.array(tD)
        
    return(procData)

def sgn(x,w, theta):
    
    '''
    Takes input vecotr x, and weight vecotor w and computes the dot product
    if the dot product is positive returns a 1, otherwise return a 0.  
    '''
    
    if   np.dot(x,w) >= theta:
        sign = 1 # Positive
    else:
        sign = 0 # Negative
    
    return(sign)

def initWeight():
    
    '''
    Initializes the weight vector for digits 0 - 9. 
    The weight vector consists of 785 elements that are randomized
    between .25 and .95
    '''
    
    # init weight list
    weight = []
    
    # randomizes 785 elements in a weight array for each digit (0-9)
    for digit in range(0,10):
        weight.append(np.ones(785))

    
#    for digit in range(0,10):
#        weight[digit] = np.append(weight[digit],785)
    
    # transforms list into numpy array
    return(np.array(weight))

def instWinnow(instance, weight, y, theta):
    
    '''
    Performs one iteration of the Perceptron Algorithm. 
    Executes multiple instances of this for each digit of the MNSIT Data. 
    '''
    
    # Returns the sign of the dot product of the instance and weight vector
    # 1 if the dot product is zero. 0 otherwise
    yprime = sgn(instance, weight,theta)
    
    # Computes the dot product of the instance and weight vector
    wSum = np.dot(instance,weight)   
    
    # if the instance sign does not equal the given gold Y value
    # then update the weights 
    if yprime != y:
        
        if y == 1: 

            for x in range(0,len(instance)):
                if instance[x] == 1:
                    weight[x] = weight[x]*1.5 # Promotion

        else:
            for x in range(0,len(instance)):
                if instance[x] == 1:
                    weight[x] = weight[x]*0.7 # Demotion


    return(wSum,weight)

def digGuess(instance, label, weight,theta):
    
    '''
    Performs the perceptron algorithm on all 10 digits in the MNSIT Data Set
    Using one instance of the data.
    Returns the updated weights and the perceptron index (0 - 9) that had the
    highest total sum.
    '''
    weightSum = np.zeros(10)
    y_gold = np.zeros(10)
    y_gold[label] = 1
        
    for digit in range(0,10):
        
        weightSum[digit], weight[digit] = instWinnow(instance, weight[digit], y_gold[digit],theta)
    
    return(weightSum.argmax(), weight)
    
def winnowTrain(data,label,weight,theta):
    
    guesses = []
    for entry in range(0,len(data)):
        
        guessIndex,weight = digGuess(data[entry], label[entry], weight,theta)
        guesses.append(guessIndex)
        
    return(guesses, weight)


def winnowTest(data,label,weight):
    
    guesses = []
    
    for entry in range(0,len(data)):
        weightSum = []
        
        for index in range(0,len(weight)):
            
            weightSum.append(np.dot(weight[index], data[entry]))
            
        guessIndex = np.argmax(weightSum)
        guesses.append(guessIndex)
        
    return(guesses)

def Accuracy(labels, guesses):

    fTotal = 0.0
    fList = []
    
    for digit in set(labels): 
        
        f1Score = f1IndScore(digit, labels, guesses)        
        fList.append(f1Score)

    fTotal = sum(fList)
    
    f1ScoreAve = float(fTotal)/(len(set(labels)))    
    return(100*round(f1ScoreAve,3))

def f1IndScore(digit, labels, guesses):
    
    T,F = 0.0,0.0
    
    for entry in range(0,len(labels)):
        if labels[entry] == digit:
            if guesses[entry] == labels[entry]:
                T = T + 1   
            else:
                F = F + 1
                
        if guesses[entry] == digit:
            if guesses[entry] != labels[entry]:
                F = F + 1
    f1 = 2*T/((2*T)+F)
    return(f1) 

def dataOut(DATA_FOLDER, trainingSize):
    
    # Changes the Directory to the directory where the Data is located
    os.chdir(DATA_FOLDER)

    # Read in Image Data for Training and Testing
    trainData = readImage('train-images-idx3-ubyte.gz')
    trainLabels = readImage('train-labels-idx1-ubyte.gz')
    testData = readImage('t10k-images-idx3-ubyte.gz')
    testLabels = readImage('t10k-labels-idx1-ubyte.gz')
    
    # Process Crude Data into flattened BITMAPS that are in the correct form 
    # and size for training 
    trainLabels = trainLabels[0:trainingSize]    
    procTrainData = dataProcess(trainData)
    procTrainData = procTrainData[0:trainingSize]
    
    # Process Crude Data into flattened BITMAPS that are in the correct form 
    # and size for Testing. 
    procTestData = dataProcess(testData)      
    
    return(procTrainData, trainLabels, procTestData, testLabels)


def shuffle(procTrainData, trainLabels):
    
    combine = list(zip(procTrainData,trainLabels))
    rand.shuffle(combine)
    procTrainData,trainLabels = zip(*combine)

    return(procTrainData,trainLabels)


'''
*******************************************************************************
'''

def tuning(procTrainData, trainLabels, procTestData, testLabels):
    '''
    This Tunes the Perceptron algorithsm using hyper-parameters: Training Size,
    number of epochs and learning rate adjustments. Due to the O(3N^3) nature
    of this function it takes an extreme amount of time to run. 
    '''

    tuneProcess = ['trainingSet','epochs', 'Theta']
    
    for entry in tuneProcess:
        
        if entry == 'trainingSet':
            epochs = 20
            theta = 785
            trainingSize = np.arange(500,10250,250)
            
            size1 = []
            f1ScoreTR1 = []
            f1ScoreTE1 = []
            
            for i in trainingSize:
                data = procTrainData[0:i]
                labels = trainLabels[0:i]
                weights = initWeight()
                
                for runs in range(0,epochs):
                    # Shuffles Data each epoch
                    data,labels = shuffle(data, labels)
                    
                    guessesTR1,weights = winnowTrain(data, labels, weights, theta)
                    
                guessesTE1 = winnowTest(procTestData, testLabels, weights)
                
                f1ScoreTR1.append(Accuracy(labels,guessesTR1))
                f1ScoreTE1.append(Accuracy(testLabels,guessesTE1))
                size1.append(i)
            
            plt.xlabel('Training Size')
            plt.ylabel('F1 Score')
            #plt.ylim(0,100)
            plt.xlim(0,10250)
            plt.xticks(range(0,11000,1000))
            #plt.yticks(np.arange(0,110,10))
            plt.autoscale(axis = 'y')
            plt.title('Learning Curve: Training Size')
            plt.plot(size1,f1ScoreTR1, color = 'blue', label = 'Training')
            plt.plot(size1,f1ScoreTE1, color = 'red', label = 'Testing')
            plt.legend()
            plt.show()
            
            
            
        if entry == 'epochs':
            epochs = np.arange(10,110,10)
            theta = 785
            trainingSize = 3000
            size2 = []
            f1ScoreTR2 = []
            f1ScoreTE2 = []
            
            
            for i in epochs:

                data = procTrainData[0:trainingSize]
                labels = trainLabels[0:trainingSize]
                #randomizes weights for each digit of interest
                weights = initWeight()
                
                for runs in range(0,i):
                    # Shuffles Data each epoch
                    data,labels = shuffle(data, labels)
                    
                    guessesTR2,weights = winnowTrain(data, labels, weights, theta)
                
                guessesTE2 = winnowTest(procTestData, testLabels, weights)
                    
                f1ScoreTR2.append(Accuracy(labels,guessesTR2))
                f1ScoreTE2.append(Accuracy(testLabels,guessesTE2))
                size2.append(i)
            
            plt.xlabel('Epochs')
            plt.ylabel('F1 Score')
            #plt.ylim(0,100)
            plt.xlim(0,110)
            plt.xticks(range(0,110,10))
            #plt.yticks(np.arange(0,110,10))
            plt.autoscale(axis = 'y')
            plt.title('Learning Curve: Epoch Number')
            plt.plot(size2,f1ScoreTR2, color = 'blue', label = 'Training')
            plt.plot(size2,f1ScoreTE2, color = 'red', label = 'Testing')
            plt.legend()
            plt.show()
            
            
        if entry == 'Theta':
            epochs = 20
            theta = np.arange(100,1100,100)
            trainingSize = 3000
            
                
            size3 = []
            f1ScoreTR3 = []
            f1ScoreTE3 = []
            
            
            for i in theta:
                data = procTrainData[0:trainingSize]
                labels = trainLabels[0:trainingSize]
                weights = initWeight()

                
                for runs in range(0,epochs):
        
                    guessesTR3,weights = winnowTrain(data, labels, weights, i)
                    
                guessesTE3 = winnowTest(procTestData, testLabels, weights)
                    
                f1ScoreTR3.append(Accuracy(labels,guessesTR3))
                f1ScoreTE3.append(Accuracy(testLabels,guessesTE3))
                size3.append(i)

            plt.xlabel('Theta')
            plt.ylabel('F1 Score')
            #plt.ylim(0,100)
            plt.xlim(0,110)
            plt.xticks(range(100,1100,100))
            #plt.yticks(np.arange(0,110,10))
            plt.autoscale(axis = 'y')
            plt.title('Learning Curve: Theta')
            plt.plot(size3,f1ScoreTR3, color = 'blue', label = 'Training')
            plt.plot(size3,f1ScoreTE3, color = 'red', label = 'Testing')
            plt.legend()
            plt.show()
    
    
    

'''
*******************************************************************************
'''

#%run vanilla_perceptron.py 10000 50 .001 C:\Users\owner\Documents\PythonScripts\DATA_FOLDER


if __name__ == "__main__":
        
    # Init input Data # 2000,20,theta = 785, p = 1.5, d = 0.8 TR = 70%
    trainingSize = 2000
    epochs = 10
    theta = 785
    path = os.getcwd()
    DATA_FOLDER = path + '/DATA_FOLDER'
    
    # Sets input function variables to the user input if all the required 
    # inputs are present. 
    if len(sys.argv) >= 5:
        trainingSize = int(sys.argv[1])
        epochs = int(sys.argv[2])
        theta = float(sys.argv[3])
        DATA_FOLDER = ''
        for entry in range(4,len(sys.argv)):
            DATA_FOLDER =  DATA_FOLDER + sys.argv[entry]
        
    # Changes the Directory to the directory where the Data is located
    os.chdir(DATA_FOLDER)
    
    # Reads Image Data, processes it and returns data that is ready for the
    # Machine Learning Algorithm
    procTrainData, trainLabels, procTestData, testLabels = dataOut(DATA_FOLDER,trainingSize) 
    
    
    # This function performs tuning on the vanilla Perceptron
    # It is commented out for submittal because data.cs.purdue.edu 
    # does not support the pyplot library and the TA's asked for them 
    # to be submitted in the report instead. When run, this function outputs 
    # the three learning curves for 'Epoch', 'Training Size' and 'Learning Rate'
    
    #tuning(procTrainData, trainLabels, procTestData, testLabels)
    
    #randomizes weights for each digit of interest
    weights = initWeight()
    
    # Train the Perceptron Algorithm using the defined function inputs
    for x in range(0,epochs):
        # Shuffles Data each epoch
        procTrainData,trainLabels = shuffle(procTrainData, trainLabels)
        
        guessesTR,weights = winnowTrain(procTrainData, trainLabels, weights,theta)
    
    # Uses the weight from the training process to make guesses for testing
    guessesTE = winnowTest(procTestData, testLabels, weights)
    
    # Training F1 Score
    F1TR = Accuracy(trainLabels,guessesTR)
    
    # Testing F1 Score
    F1TE = Accuracy(testLabels,guessesTE)
    
    # Returns back to Home Directory
    os.chdir(path)
    
    # Prints out Training and Testing F1 Score
    print('Training F1 Score: ' + str(F1TR))
    print('Testing F1 Score: ' + str(F1TE))
    




