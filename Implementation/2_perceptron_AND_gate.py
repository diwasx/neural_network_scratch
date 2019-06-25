#!/bin/python3

import sys
sys.path.append('..')
# import perceptron
from perceptron import Perceptron

inputLen = 3
learnRate = 0.1
bias = 1
# This can be used if only perceptron module is imported
# p = perceptron.Perceptron(inputLen)
p = Perceptron(inputLen)

for i in range(20):
    inputs = [bias,0,0]
    target = 0
    p.train(inputs,target, learnRate)
    inputs = [bias,0,1]
    target = 0
    p.train(inputs,target, learnRate)
    inputs = [bias,1,0]
    target = 0
    p.train(inputs,target, learnRate)
    inputs = [bias,1,1]
    target = 1
    p.train(inputs,target, learnRate)

print("After training => " + str(p.guess([bias,0,0])))
print("After training => " + str(p.guess([bias,0,1])))
print("After training => " + str(p.guess([bias,1,0])))
print("After training => " + str(p.guess([bias,1,1])))
