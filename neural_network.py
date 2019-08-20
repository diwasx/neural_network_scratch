#!/bin/python3
# Two layer neural network (Fully connected)

import random
import math
import sys
import numpy as np
from tkinter import * 
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Function return value between 0 and 1 for all element of matrix
def sigmoid(x):
    tmpMatrix = np.zeros(shape=(len(x),1))
    # f(x) = 1/(1+e^-x)
    for i in range(len(x)):
        # Argument is a large negative value, so it is calling exp() with a large positive value. It is very easy to exceed floating point range that way
        # This condition solve that problem
        if (x[i][0]) < 0:
            tmpMatrix[i][0] = 1 - 1/(1+math.exp(x[i][0]))
        else:
            # Original sigmoid function
            tmpMatrix[i][0] = 1/(1+math.exp(-x[i][0]))
    # print(tmpMatrix)
    return (tmpMatrix)

# Function return derivative value of sigmoid function for all element of matrix
def dSigmoid(x):
    tmpMatrix = np.zeros(shape=(len(x),1))
    # sig'(x) = sig(x)[1-sig(x)]
    for i in range(len(x)):
        tmpMatrix[i][0] = x[i][0] * (1 - x[i][0])
    return (tmpMatrix)

class NeuralNetwork:

    # Constructor
    def __init__(self, inputLen, hiddenLen, outputLen):
        self.i = inputLen # input
        self.j = hiddenLen # hidden
        self.k = outputLen # output


        self.scaleFac = 1
        self.gapVal = 200
        # Creating weights and assiging random values
        # Weights for Hidden
        self.weightsHid = np.zeros(shape=(self.j,self.i)) # w
        for m in range(self.j):
            for n in range(self.i):
                randVal = random.uniform(-1,1)
                self.weightsHid[m][n] =  randVal
        # print("\nWeights of hidden\n" + str(self.weightsHid))

        # Weights for Output
        self.weightsOut = np.zeros(shape=(self.k,self.j)) # w'
        for m in range(self.k):
            for n in range(self.j):
                randVal = random.uniform(-1,1)
                self.weightsOut[m][n] =  randVal
        # print("\nWeights of output\n" + str(self.weightsOut))

        # Creating Bias and assiging random values
        # Bias for Hidden
        self.biasHid = np.zeros(shape=(self.j,1))
        for j in range(self.j):
            randVal = random.uniform(-1,1)
            self.biasHid[j][0] =  randVal
        # print("\nBias of hidden\n" + str(self.biasHid))

        # Bias for Output
        self.biasOut = np.zeros(shape=(self.k,1))
        for k in range(self.k):
            randVal = random.uniform(-1,1)
            self.biasOut[k][0] =  randVal
        # print("\nBias of output\n" + str(self.biasOut))
        # print("\n")

    # Algorithm that computes output based on weight and bias (similar to guess function in perceptron)
    def feedForward(self,inputs):
        # print("\n<-- Feed forward --->")
        self.inputs = inputs
        # print("\nInput matrix\n" + str(self.inputs))

        self.hiddens = self.weightsHid.dot(self.inputs) + self.biasHid
        # print("\nMatrix of Hiddens \n"+ str(self.hiddens))
        self.hiddens = sigmoid(self.hiddens)
        # print("\nMatrix of Hiddens after activation\n"+ str(self.hiddens))

        self.outputs = self.weightsOut.dot(self.hiddens) + self.biasOut
        # print("\nMatrix of Outputs \n" + str(self.outputs))
        self.outputs = sigmoid(self.outputs)
        # print(bcolors.OKBLUE + "\nMatrix of Outputs after activation\n"+ str(self.outputs) + bcolors.ENDC)
        return self.outputs

    # Training NN using Supervised learning
    def trainSVLearing(self, inputs, targets, learningR):
        # Guess outputs from inputs
        self.feedForward(inputs)

        # print("\n<-- Backpropagation --->")

        # Calculate output errors
        output_errors = targets - self.outputs
        # print("\nMatrix of Inputs\n"+ str(inputs))
        # print("\nMatrix of Targets\n"+ str(targets))
        # print("\nMatrix of Outputs \n"+ str(self.outputs))
        # print("\nMatrix of Output Errors\n"+ str(output_errors))

        # print("\n<-- Gradient Descent of Output--->")
        # Output Gradient
        output_gradient = dSigmoid(self.outputs)
        # print("\nOutput Gradient\n"+ str(output_gradient))

        # Delta Output weights
        # weightsOut_delta = learningR * output_errors * output_gradient ∙ hiddens.transpose()
        # * represent hadamard product dot(∙) represent matrix dot product

        tmpOut = learningR * output_errors * output_gradient

        # print("\ntmpOut\n"+ str(tmpOut))

        # print("\nHiddens Transpose\n"+ str(self.hiddens.transpose()))
        weightsOut_delta = tmpOut.dot(self.hiddens.transpose())
        # print("\nOutput deltaWeight\n"+ str(weightsOut_delta))

        # New Output weights
        self.weightsOut += weightsOut_delta
        # print(bcolors.OKBLUE + "\nNew Output Weights\n"+ str(self.weightsOut))

        # new biasOut += learningR * output_errors * output_gradient
        self.biasOut+=tmpOut
        # print("\nNew Output Bias\n"+ str(self.biasOut) + bcolors.ENDC)


        # Calculate hidden errors
        weightsOutTranspose = self.weightsOut.transpose()
        # print("\nWeight of outputs transpose\n"+ str(weightsOutTranspose))
        hidden_errors = weightsOutTranspose.dot(output_errors)
        # print("\nMatrix of Hidden Errors\n"+ str(hidden_errors))

        # print("\n<-- Gradient Descent of Hidden--->")
        # Hidden Gradient
        hidden_gradient = dSigmoid(self.hiddens)
        # print("\nHiddens Gradient\n"+ str(hidden_gradient))

        # Delta Hidden weights
        # weightsHid_delta = learningR * hidden_errors * hidden_gradient ∙ inputs.transpose()
        # (*) represent hadamard product dot(∙) represent matrix dot product

        tmpHid = learningR * hidden_errors * hidden_gradient 

        # print("\ntmpHid\n"+ str(tmpHid))

        # print("\nInputs Transpose\n"+ str(self.inputs.transpose()))
        weightsHid_delta = tmpHid.dot(self.inputs.transpose())
        # print("\nHidden deltaWeight\n"+ str(weightsHid_delta))

        # New Hiddens weights
        self.weightsHid += weightsHid_delta
        # print(bcolors.OKBLUE + "\nNew Hiddens Weights\n"+ str(self.weightsHid))

        # new biasHid += learningR * hidden_errors * hidden_gradient
        self.biasHid+=tmpHid
        # print("\nNew Hiddens Bias\n"+ str(self.biasHid) + bcolors.ENDC)

    def nnVisualization(self):
        # Best view upto 18 nodes
        tk = Tk()
        widthSize = 800
        # heightSize = 600
        heightSize = 700
        frameRate = 60
        frameSpeed = int(1 / frameRate * 1000)

        canvas = Canvas(tk, width=widthSize, height=heightSize)
        tk.title("Drawing_float")
        canvas.pack()

        # For input layer
        inputNodes = [None] * self.i
        hiddenWLines = [None] * self.i
        y = 0
        # Starting Position
        yStart = 50 * self.scaleFac
        for m in range(self.i):

            # If object is out of canvas, small scale factor
            if(y >= heightSize):
                self.scaleFac = self.scaleFac * 0.999
                self.gapVal -= 0.8
                tk.destroy()
                return self.nnVisualization()
            
            x1, y1 = 50/self.scaleFac, yStart+y
            x2, y2 = 90/self.scaleFac, yStart+y+40
            inputNodes[m] = canvas.create_oval(x1*self.scaleFac, y1*self.scaleFac, x2*self.scaleFac, y2*self.scaleFac, fill="white")

            # For hidden weight line
            xic, yic = (x1+x2)/2, (y1+y2)/2
            yTmp = 0
            yStartTmp = 50 * self.scaleFac
            for n in range(self.j):
                x1, y1 = 350/self.scaleFac, yStartTmp+yTmp
                x2, y2 = 390/self.scaleFac, yStartTmp+yTmp+40
                xhc, yhc = (x1+x2)/2, (y1+y2)/2
                hiddenWLines = canvas.create_line(xic*self.scaleFac, yic*self.scaleFac, xhc*self.scaleFac, yhc*self.scaleFac, fill="white")

                # GapsTmp
                yTmp = yTmp  + self.gapVal

            # Gaps
            y = y + self.gapVal

        # For Hidden layer
        hiddenNodes = [None] * self.j
        y = 0
        yStart = 50 * self.scaleFac
        for m in range(self.j):

            # If object is out of canvas, small scale factor
            if(y >= heightSize):
                self.scaleFac = self.scaleFac * 0.999
                self.gapVal -= 0.8
                tk.destroy()
                return self.nnVisualization()

            x1, y1 = 350/self.scaleFac, yStart+y
            x2, y2 = 390/self.scaleFac, yStart+y+40
            hiddenNodes[m] = canvas.create_oval(x1* self.scaleFac, y1* self.scaleFac, x2* self.scaleFac, y2* self.scaleFac, fill="white")

            # For output weight line
            xhc, yhc = (x1+x2)/2, (y1+y2)/2
            yTmp = 0
            yStartTmp = 50* self.scaleFac
            for n in range(self.k):
                x1, y1 = 650/self.scaleFac, yStartTmp+yTmp
                x2, y2 = 690/self.scaleFac, yStartTmp+yTmp+40
                xoc, yoc = (x1+x2)/2, (y1+y2)/2
                hiddenWLines = canvas.create_line(xhc* self.scaleFac, yhc* self.scaleFac, xoc* self.scaleFac, yoc* self.scaleFac, fill="white")

                # GapsTmp
                yTmp = yTmp  + self.gapVal

            # Gaps
            y = y  + self.gapVal

        # For Output layer
        outputNodes = [None] * self.k
        y = 0
        yStart = 50* self.scaleFac
        for m in range(self.k):

            # If object is out of canvas, small scale factor
            if(y >= heightSize):
                self.scaleFac = self.scaleFac * 0.999
                self.gapVal -= 0.8
                tk.destroy()
                return self.nnVisualization()

            x1, y1 = 650/self.scaleFac, yStart+y
            x2, y2 = 690/self.scaleFac, yStart+y+40
            outputNodes[m] = canvas.create_oval(x1* self.scaleFac, y1* self.scaleFac, x2* self.scaleFac, y2* self.scaleFac, fill="white")

            # Gaps
            y = y  + self.gapVal

        while True:
            tk.after(frameSpeed, tk.update()) 
        tk.mainloop()


    # For Neuro Evolution
    def copy(self):
        return (self)

    def mutate(self, mutationRate):

        def mutateElement(val):
            if (random.uniform(0,1) < mutationRate):
                # x = (random.uniform(0,100))
                x = 2 * random.uniform(0,1) - 1
                return x
            else:
                return val
            
        vfunc = np.vectorize(mutateElement)
        self.weightsHid = vfunc(self.weightsHid)
        self.weightsOut = vfunc(self.weightsOut)
        self.biasHid = vfunc(self.biasHid)
        self.biasOut = vfunc(self.biasOut)
