#!/bin/python3

import sys
# sys.path.append('../perceptron.py')
# sys.path.append("..") # Adds higher directory to python modules path.
# import perceptron
from perceptron import Perceptron

inputLen = 2
learnRate = 0.1
p = Perceptron(inputLen)

for i in range(10):
    inputs = [0,0]
    target = 0
    p.train(inputs,target, learnRate)
    inputs = [0,1]
    target = 1
    p.train(inputs,target, learnRate)
    inputs = [1,0]
    target = 1
    p.train(inputs,target, learnRate)
    inputs = [1,1]
    target = 1
    p.train(inputs,target, learnRate)

print("After training => " + str(p.guess([0,0])))
print("After training => " + str(p.guess([0,1])))
print("After training => " + str(p.guess([1,0])))
print("After training => " + str(p.guess([1,1])))

