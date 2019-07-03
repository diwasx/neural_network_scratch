#!/bin/python3
# XOR and XNOR cannot be linearly separated, so perceptron cannot give correct result

import sys
import random
import numpy as np
sys.path.append('..')
from neural_network import NeuralNetwork

inputLen = 2
hiddenLen = 100
outputLen = 1
learningRate = 0.2
n = NeuralNetwork(inputLen, hiddenLen, outputLen)
# n = NeuralNetwork(2, 2, 1)

training_data = {
    1: {'inputs': np.array([[0],[0]]), 'targets': np.array([[0]])},
    2: {'inputs': np.array([[0],[1]]), 'targets': np.array([[1]])},
    3: {'inputs': np.array([[1],[0]]), 'targets': np.array([[1]])},
    4: {'inputs': np.array([[1],[1]]), 'targets': np.array([[0]])},
    }

# x = random.choice(list(training_data.values()))
# print (x.get('inputs'))

for i in range(1000):
    # x = random.choice(list(training_data.keys()))
    x = random.choice(list(training_data.values()))
    inputs =  x.get('inputs')
    targets =  x.get('targets')
    # print("Input = " + str(inputs))
    # print("Targets = " + str(targets))
    n.trainSVLearing(inputs,targets,learningRate)


inputs = np.array([
    [0],
    [0]
    ])
n.feedForward(inputs)

inputs = np.array([
    [0],
    [1]
    ])
n.feedForward(inputs)

inputs = np.array([
    [1],
    [0]
    ])
n.feedForward(inputs)

inputs = np.array([
    [1],
    [1]
    ])
n.feedForward(inputs)
