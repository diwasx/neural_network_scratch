#!/bin/python3
# Two layer neural network (Fully connected)

import random
import math
import sys
from tkinter import * 
import numpy as np

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
        self.tk = Tk()
        self.tk.title("Drawing_float")
        widthSize = 800
        heightSize = 700
        self.canvas = Canvas(self.tk, width=widthSize, height=heightSize)

        # Creating weights and assiging random values
        # Weights for Hidden
        self.weightsHid = np.zeros(shape=(self.j,self.i)) # w
        for m in range(self.j):
            for n in range(self.i):
                randVal = random.uniform(-1,1)
                self.weightsHid[m][n] =  randVal

        # Weights for Output
        self.weightsOut = np.zeros(shape=(self.k,self.j)) # w'
        for m in range(self.k):
            for n in range(self.j):
                randVal = random.uniform(-1,1)
                self.weightsOut[m][n] =  randVal

        # Creating Bias and assiging random values
        # Bias for Hidden
        self.biasHid = np.zeros(shape=(self.j,1))
        for j in range(self.j):
            randVal = random.uniform(-1,1)
            self.biasHid[j][0] =  randVal

        # Bias for Output
        self.biasOut = np.zeros(shape=(self.k,1))
        for k in range(self.k):
            randVal = random.uniform(-1,1)
            self.biasOut[k][0] =  randVal

    # Algorithm that computes output based on weight and bias (similar to guess function in perceptron)
    def feedForward(self,inputs):
        self.inputs = inputs

        self.hiddens = self.weightsHid.dot(self.inputs) + self.biasHid
        self.hiddens = sigmoid(self.hiddens)

        self.outputs = self.weightsOut.dot(self.hiddens) + self.biasOut
        self.outputs = sigmoid(self.outputs)
        return self.outputs

    # Training NN using Supervised learning
    def trainSVLearing(self, inputs, targets, learningR):
        # Guess outputs from inputs
        self.feedForward(inputs)


        # Calculate output errors
        output_errors = targets - self.outputs

        # Output Gradient
        output_gradient = dSigmoid(self.outputs)

        # Delta Output weights
        # weightsOut_delta = learningR * output_errors * output_gradient ∙ hiddens.transpose()
        # * represent hadamard product dot(∙) represent matrix dot product

        tmpOut = learningR * output_errors * output_gradient


        weightsOut_delta = tmpOut.dot(self.hiddens.transpose())

        # New Output weights
        self.weightsOut += weightsOut_delta

        # new biasOut += learningR * output_errors * output_gradient
        self.biasOut+=tmpOut


        # Calculate hidden errors
        weightsOutTranspose = self.weightsOut.transpose()
        hidden_errors = weightsOutTranspose.dot(output_errors)

        # Hidden Gradient
        hidden_gradient = dSigmoid(self.hiddens)

        # Delta Hidden weights
        # weightsHid_delta = learningR * hidden_errors * hidden_gradient ∙ inputs.transpose()
        # (*) represent hadamard product dot(∙) represent matrix dot product

        tmpHid = learningR * hidden_errors * hidden_gradient 


        weightsHid_delta = tmpHid.dot(self.inputs.transpose())

        # New Hiddens weights
        self.weightsHid += weightsHid_delta

        # new biasHid += learningR * hidden_errors * hidden_gradient
        self.biasHid+=tmpHid

    def nnStructure(self):
        # Best view upto 18 nodes
        frameRate = 60
        frameSpeed = int(1 / frameRate * 1000)
        widthSize = 800
        heightSize = 700
        self.canvas.pack()
                
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
                self.canvas.delete("all")
                return self.nnStructure()
            
            x1, y1 = 50/self.scaleFac, yStart+y
            x2, y2 = 90/self.scaleFac, yStart+y+40
            inputNodes[m] = self.canvas.create_oval(x1*self.scaleFac, y1*self.scaleFac, x2*self.scaleFac, y2*self.scaleFac, fill="white")

            # For hidden weight line
            xic, yic = (x1+x2)/2, (y1+y2)/2
            yTmp = 0
            yStartTmp = 50 * self.scaleFac
            for n in range(self.j):
                x1, y1 = 350/self.scaleFac, yStartTmp+yTmp
                x2, y2 = 390/self.scaleFac, yStartTmp+yTmp+40
                xhc, yhc = (x1+x2)/2, (y1+y2)/2
                weight = self.weightsHid[n][m]
                if(weight > 0):
                    color = "red";
                else:
                    color = "green";
                hiddenWLines = self.canvas.create_line(xic*self.scaleFac, yic*self.scaleFac, xhc*self.scaleFac, yhc*self.scaleFac, fill=color, width=3)
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
                self.canvas.delete("all")
                return self.nnStructure()

            x1, y1 = 350/self.scaleFac, yStart+y
            x2, y2 = 390/self.scaleFac, yStart+y+40
            hiddenNodes[m] = self.canvas.create_oval(x1* self.scaleFac, y1* self.scaleFac, x2* self.scaleFac, y2* self.scaleFac, fill="white")

            # For output weight line
            xhc, yhc = (x1+x2)/2, (y1+y2)/2
            yTmp = 0
            yStartTmp = 50* self.scaleFac
            for n in range(self.k):
                x1, y1 = 650/self.scaleFac, yStartTmp+yTmp
                x2, y2 = 690/self.scaleFac, yStartTmp+yTmp+40
                xoc, yoc = (x1+x2)/2, (y1+y2)/2
                weight = self.weightsOut[n][m]
                if(weight > 0):
                    color = "red";
                else:
                    color = "green";
                hiddenWLines = self.canvas.create_line(xhc* self.scaleFac, yhc* self.scaleFac, xoc* self.scaleFac, yoc* self.scaleFac, fill=color, width=3)
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
                self.canvas.delete("all")
                return self.nnStructure()

            x1, y1 = 650/self.scaleFac, yStart+y
            x2, y2 = 690/self.scaleFac, yStart+y+40
            outputNodes[m] = self.canvas.create_oval(x1* self.scaleFac, y1* self.scaleFac, x2* self.scaleFac, y2* self.scaleFac, fill="white")
            # Gaps
            y = y  + self.gapVal

        # while True:
        self.tk.after(frameSpeed, self.tk.update()) 
        return (self.tk)

    def trainSVLearingVisualization(self,inputs,targets,learningRate):
        # Best view upto 18 nodes
        self.canvas.pack()
        widthSize = 800
        heightSize = 700
        frameRate = 60
        frameSpeed = int(1 / frameRate * 1000)

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
                self.canvas.delete("all")
                return self.trainSVLearingVisualization(inputs,targets,learningRate)
            
            x1, y1 = 50/self.scaleFac, yStart+y
            x2, y2 = 90/self.scaleFac, yStart+y+40
            inputNodes[m] = self.canvas.create_oval(x1*self.scaleFac, y1*self.scaleFac, x2*self.scaleFac, y2*self.scaleFac, fill="white")

            # For hidden weight line
            xic, yic = (x1+x2)/2, (y1+y2)/2
            yTmp = 0
            yStartTmp = 50 * self.scaleFac
            for n in range(self.j):
                x1, y1 = 350/self.scaleFac, yStartTmp+yTmp
                x2, y2 = 390/self.scaleFac, yStartTmp+yTmp+40
                xhc, yhc = (x1+x2)/2, (y1+y2)/2
                weight = self.weightsHid[n][m]
                if(weight > 0):
                    color = "red";
                else:
                    color = "green";

                hiddenWLines = self.canvas.create_line(xic*self.scaleFac, yic*self.scaleFac, xhc*self.scaleFac, yhc*self.scaleFac, fill=color, width=3)
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
                self.canvas.delete("all")
                return self.trainSVLearingVisualization(inputs,targets,learningRate)

            x1, y1 = 350/self.scaleFac, yStart+y
            x2, y2 = 390/self.scaleFac, yStart+y+40
            hiddenNodes[m] = self.canvas.create_oval(x1* self.scaleFac, y1* self.scaleFac, x2* self.scaleFac, y2* self.scaleFac, fill="white")

            # For output weight line
            xhc, yhc = (x1+x2)/2, (y1+y2)/2
            yTmp = 0
            yStartTmp = 50* self.scaleFac
            for n in range(self.k):
                x1, y1 = 650/self.scaleFac, yStartTmp+yTmp
                x2, y2 = 690/self.scaleFac, yStartTmp+yTmp+40
                xoc, yoc = (x1+x2)/2, (y1+y2)/2
                weight = self.weightsOut[n][m]
                if(weight > 0):
                    color = "red";
                else:
                    color = "green";
                hiddenWLines = self.canvas.create_line(xhc* self.scaleFac, yhc* self.scaleFac, xoc* self.scaleFac, yoc* self.scaleFac, fill=color, width=3)
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
                self.canvas.delete("all")
                return self.trainSVLearingVisualization(inputs,targets,learningRate)

            x1, y1 = 650/self.scaleFac, yStart+y
            x2, y2 = 690/self.scaleFac, yStart+y+40
            outputNodes[m] = self.canvas.create_oval(x1* self.scaleFac, y1* self.scaleFac, x2* self.scaleFac, y2* self.scaleFac, fill="white")
            # Gaps
            y = y  + self.gapVal

        self.tk.after(frameSpeed, self.tk.update()) 
        self.trainSVLearing(inputs,targets,learningRate)
        return (self.tk)

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
