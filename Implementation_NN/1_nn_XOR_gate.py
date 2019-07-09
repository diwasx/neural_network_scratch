#!/bin/python3

import sys
import os
import random
import numpy as np
sys.path.append('..')
# from neural_network import NeuralNetwork
from neural_network_verbose import NeuralNetwork

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

inputLen = 2
hiddenLen = 4
outputLen = 1
learningRate = 0.1

n = NeuralNetwork(inputLen, hiddenLen, outputLen)
# n = NeuralNetwork(2, 2, 1)

training_data = {
    1: {'inputs': np.array([[0],[0]]), 'targets': np.array([[0]])},
    2: {'inputs': np.array([[0],[1]]), 'targets': np.array([[1]])},
    3: {'inputs': np.array([[1],[0]]), 'targets': np.array([[1]])},
    4: {'inputs': np.array([[1],[1]]), 'targets': np.array([[0]])},
    }

print("\033[4m" + "\n### Training ###" + "\033[0m")
# x = random.choice(list(training_data.values()))
# print (x.get('inputs'))

for i in range(10000):
# for i in range(8000):
    # x = random.choice(list(training_data.keys()))
    x = random.choice(list(training_data.values()))
    inputs =  x.get('inputs')
    targets =  x.get('targets')
    # print("Input = " + str(inputs))
    # print("Targets = " + str(targets) + "\n")
    blockPrint()
    n.trainSVLearing(inputs,targets,learningRate)


enablePrint()
print("\033[4m" + "\n### Testing Phase ###" + "\033[0m")
inputs = np.array([
    [0],
    [0]
    ])
print(n.feedForward(inputs))

inputs = np.array([
    [0],
    [1]
    ])
print(n.feedForward(inputs))

inputs = np.array([
    [1],
    [0]
    ])
print(n.feedForward(inputs))

inputs = np.array([
    [1],
    [1]
    ])
print(n.feedForward(inputs))
