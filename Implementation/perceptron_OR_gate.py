#!/bin/python3

import sys
sys.path.append('..')
# import perceptron
from perceptron import Perceptron

inputLen = 2
learnRate = 0.1
# This can be used if only perceptron module is imported
# p = perceptron.Perceptron(inputLen)
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

