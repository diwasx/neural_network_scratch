#!/bin/python3

import random

# Function return 1 for positive and -1 for negative value
def sign(a):
    if (a > 0):
        return 1
    elif (a <= 0):
        return -1

class Perceptron:

    # Constructor
    def __init__(self,inputLen):
        self.weights = [None] * inputLen
        self.inputs = []
        self.learnRate = 0;

        # Assiging random values to weights
        for i in range(len(self.weights)):
        # for i in self.weights:  # (similar to for each)
            # randVal = randint(0,9)
            randVal = random.uniform(-1,1)
            self.weights[i] =  randVal
        print("Random weights = " + str(self.weights))
    
    # Function for guessing output
    def guess(self,inputs):
        self.inputs = inputs
        total = 0;
        for i in range(len(self.weights)):
            total += self.inputs[i]*self.weights[i];
        print("Sum = " + str(total))

        # Activation function
        output = sign(total)
        print("Output = " + str(output))
        if (output == -1):
            return 0
        else:
            return 1
    
    # Training perceptron with known answer (supervised learning)
    def train(self, inputs, target, r):
        self.learnRate = r
        self.inputs = inputs
        guessVal = self.guess(self.inputs)
        print("Inputs value = " + str(inputs))
        print("Target value = " + str(target))
        print("Guess value = " + str(guessVal))
        error = target - guessVal
        print("Error = " + str(error))

        # Tuning weights
        for i in range(len(self.weights)):
            # Gradient descent
            deltaWeight = error * self.inputs[i] * self.learnRate
            print("deltaWeight [" + str(i) + "] = " + str(deltaWeight))
            self.weights[i] += deltaWeight
            print("New weight [" + str(i) + "] = " + str(self.weights))
        print("Succesfully trained\n")

