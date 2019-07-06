#!/bin/python3

import sys
import random
import numpy as np
# from sklearn import datasets as d
import sklearn.datasets as d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
sys.path.append('..')
from neural_network import NeuralNetwork


learningRate = 0.1
# nn = NeuralNetwork(64, 40, 10)
nn = NeuralNetwork(64, 100, 10)
digits = d.load_digits()

print ("\nTraining -------\n")
for i in range (10):
    # (total no of data = 1797)
    # for i in range(len(digits.data)):
    for i in range(1437):
        imgTrain = digits.images[i]
        target = digits.target[i]
        # print(imgTrain)
        # print(target)

        # # Diplaying image
        # style.use('fivethirtyeight')
        # fig = plt.figure(1,figsize=(5,5))
        # fig = plt.gcf()
        # fig.canvas.set_window_title('Drawing_float')
        # plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
        # # plt.imshow(img, interpolation='nearest')
        # plt.show()

        # Training NN with training images
        imgVec = np.zeros(shape=(64,1))
        targetVec = np.zeros(shape=(10,1))
        targetVec[target] = 1
        n = 0
        for i in range(8):
            for j in range(8):
                imgVec[n] = imgTrain[i][j]
                n+=1
        # print ("\nImage Vector is \n" + str(imgVec) + "\n")
        # print ("\nTarget Vector is \n" + str(targetVec) + "\n")
        nn.trainSVLearing(imgVec, targetVec, learningRate)

# Testing NN with testing images
a = 0
start=1437
end=1797
# start=1
# end=1437
print ("\nTesting")
for z in range(start,end):
    imgTest = digits.images[z]
    imgVec = np.zeros(shape=(64,1))
    n = 0
    for i in range(8):
        for j in range(8):
            imgVec[n] = imgTest[i][j]
            n+=1
    outputMat = nn.feedForward(imgVec)
    maxVal = outputMat.max()
    output = outputMat.tolist().index(maxVal)
    target = digits.target[z]
    # print ("\nOutput Matrix\n" + str(outputMat) + "\n")
    # print ("\nGuess Value \n" + str(output) + "\n")
    # print ("\nActual Value\n" + str(target) + "\n")

    if (output == target):
        a+=1
accuracy = a/(end-start)
print ("Accuracy %\n" + str(accuracy) + "\n")
