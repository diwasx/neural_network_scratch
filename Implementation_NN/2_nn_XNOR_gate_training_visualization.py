#!/bin/python3

import sys
import random
import numpy as np
sys.path.append('..')
from neural_network import NeuralNetwork

inputLen = 2
hiddenLen = 7
outputLen = 1
learningRate = 0.1
n = NeuralNetwork(inputLen, hiddenLen, outputLen)

training_data = {
    1: {'inputs': np.array([[0],[0]]), 'targets': np.array([[1]])},
    2: {'inputs': np.array([[0],[1]]), 'targets': np.array([[0]])},
    3: {'inputs': np.array([[1],[0]]), 'targets': np.array([[0]])},
    4: {'inputs': np.array([[1],[1]]), 'targets': np.array([[1]])},
    }

# Supervised Training with Visualization
loopForVisualization = 300
loopForSpeedUp = 30
for i in range(loopForVisualization):
    print("Training no: " + str(i+1))
    for j in range (loopForSpeedUp):
        x = random.choice(list(training_data.values()))
        inputs =  x.get('inputs')
        targets =  x.get('targets')
        n.trainSVLearing(inputs,targets,learningRate)

    x = random.choice(list(training_data.values()))
    inputs =  x.get('inputs')
    targets =  x.get('targets')
    tk = n.trainSVLearingVisualization(inputs,targets,learningRate)

# Testing Part
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

tk.mainloop()
