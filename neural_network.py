#!/bin/python3
# Two layer neural network (Fully connected)

import random
import math
import sys
import numpy as np

# Function return 1 for positive and -1 for negative value
def sign(a):
    if (a > 0):
        return 1
    elif (a <= 0):
        return -1

# Function return value between 0 and 1 for x input
def sigmoid(x):
    tmpMatrix = np.zeros(shape=(len(x),1))
    # f(x) = 1/(1+e^-x)
    for i in range(len(x)):
        tmpMatrix[i][0] = (1/(1+math.exp(-x[i][0])))
    # print(tmpMatrix)
    return (tmpMatrix)

class NeuralNetwork:

    # Constructor
    def __init__(self, inputLen, hiddenLen, outputLen):
        self.i = inputLen # i
        self.j = hiddenLen # j
        self.k = outputLen # k

        # Creating weights and assiging random values
        # Weights for Hidden
        self.weightsHid = np.zeros(shape=(self.j,self.i)) # w
        for m in range(self.j):
            for n in range(self.i):
                randVal = random.uniform(-1,1)
                self.weightsHid[m][n] =  randVal
        print("\nWeights of hidden\n" + str(self.weightsHid))

        # Weights for Output
        self.weightsOut = np.zeros(shape=(self.k,self.j)) # w'
        for m in range(self.k):
            for n in range(self.j):
                randVal = random.uniform(-1,1)
                self.weightsOut[m][n] =  randVal
        print("\nWeights of output\n" + str(self.weightsOut))

        # Creating Bias and assiging random values
        # Bias for Hidden
        self.biasHid = np.zeros(shape=(self.j,1))
        for j in range(self.j):
            randVal = random.uniform(-1,1)
            self.biasHid[j][0] =  randVal
        print("\nBias of hidden\n" + str(self.biasHid))

        # Bias for Output
        self.biasOut = np.zeros(shape=(self.k,1))
        for k in range(self.k):
            randVal = random.uniform(-1,1)
            self.biasOut[k][0] =  randVal
        print("\nBias of output\n" + str(self.biasOut))

    def feedForward(self,inputs):
        print("\n<-- Feed forward --->")
        self.inputs = inputs
        print("\nInput matrix\n" + str(self.inputs))

        self.hiddens = self.weightsHid.dot(self.inputs) + self.biasHid
        print("\nMatrix of Hiddens \n"+ str(self.hiddens))
        self.hiddens = sigmoid(self.hiddens)
        print("\nMatrix of Hiddens after activation\n"+ str(self.hiddens))

        self.outputs = self.weightsOut.dot(self.hiddens) + self.biasOut
        print("\nMatrix of Outputs \n" + str(self.outputs))
        self.outputs = sigmoid(self.outputs)
        print("\nMatrix of Outputs after activation\n"+ str(self.outputs))
