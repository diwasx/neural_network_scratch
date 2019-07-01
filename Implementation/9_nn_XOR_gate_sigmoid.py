#!/bin/python3
# XOR and XNOR cannot be linearly separated, so perceptron cannot give correct result

import sys
import numpy as np
sys.path.append('..')
from neural_network import NeuralNetwork

inputLen = 2
hiddenLen = 2
outputLen = 1
input = np.array([
    [0],
    [1]
    ])

# print (input)

# n = NeuralNetwork(inputLen, hiddenLen, outputLen)
n = NeuralNetwork(2, 4, 3)
n.feedForward(input)
