#!/bin/python3

import sys
import random
import numpy as np
from tkinter import *
# from sklearn import datasets as d
import sklearn.datasets as d
import matplotlib.pyplot as plt
from matplotlib import style
from PIL import ImageTk, Image, ImageDraw, ImageOps
import PIL
sys.path.append('..')
from neural_network import NeuralNetwork

learningRate = 0.1
# nn = NeuralNetwork(64, 100, 10)
nn = NeuralNetwork(64, 50, 10)
# nn = NeuralNetwork(64, 64, 10)
# nn = NeuralNetwork(18, 18, 18)
digits = d.load_digits()

def normalization(x):
    minX = 0
    maxX = 16
    y = (x-minX)/(maxX-minX)
    y = y*255
    return int(y)

def showPic(i):
    imgTrain = digits.images[i]
    target = digits.target[i]
    # print(imgTrain)
    # print(target)

    # Diplaying image
    style.use('fivethirtyeight')
    fig = plt.figure(1,figsize=(5,6))
    fig = plt.gcf()
    plt.xlabel("Target = " + str(target))
    fig.canvas.set_window_title('Drawing_float')
    # print(imgTrain)

    vfunc = np.vectorize(normalization)
    normalizedImg = vfunc(imgTrain)
    normalizedImgInv = 255-normalizedImg
    # print(normalizedImg)
    print(normalizedImgInv)
    # plt.imshow(normalizedImg, cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.imshow(normalizedImg, cmap=plt.cm.gray, interpolation='nearest')
    plt.imshow(normalizedImgInv, cmap=plt.cm.gray, interpolation='nearest')
    # plt.imshow(normalizedImg)
    plt.show()

def training():
    print ("\nTraining -------\n")
    # Supervised Training with Visualization
    loopForSpeedUp = 10
    loopForVisualization = 40*loopForSpeedUp
    p=0
    for j in range (loopForSpeedUp):
        for k in range(1437):
            imgTrain = digits.images[k]
            target = digits.target[k]
            imgVec = np.zeros(shape=(64,1))
            targetVec = np.zeros(shape=(10,1))
            targetVec[target] = 1
            n = 0
            for i in range(8):
                for j in range(8):
                    imgVec[n] = imgTrain[i][j]
                    n+=1
            nn.trainSVLearing(imgVec,targetVec,learningRate)
            if (k%loopForVisualization == 0):
                print("Training visualization no: " + str(p+1))
                p+=1
                k = random.randint(0,1437)
                imgTrain = digits.images[i]
                target = digits.target[i]
                imgVec = np.zeros(shape=(64,1))
                targetVec = np.zeros(shape=(10,1))
                targetVec[target] = 1
                n = 0
                for i in range(8):
                    for j in range(8):
                        imgVec[n] = imgTrain[i][j]
                        n+=1
                global tkVis
                tkVis = nn.trainSVLearingVisualization(imgVec,targetVec,learningRate)

def testing():
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

def drawingCanvas():

    width = 100
    height = 100
    # width = 400
    # height = 400
    white = (255, 255, 255)
    green = (0,128,0)

    def save():
        filename = "image.png"
        image1.save(filename)
        original_image = Image.open("image.png")
        size = (8,8)
        resizeImg = ImageOps.fit(original_image, size, Image.ANTIALIAS).convert('L')
        # print (resizeImg)

        # img = Image.open('image.png')
        # img = Image.open('image.png').convert('LA')
        arr = np.array(resizeImg)
        normalizeArr = 255-arr
        # print(arr)
        # print(normalizeArr)

        imgVec = np.zeros(shape=(64,1))
        n = 0
        for i in range(8):
            for j in range(8):
                imgVec[n] = normalizeArr[i][j]
                n+=1
        # print(imgVec)
        outputMat = nn.feedForward(imgVec)
        maxVal = outputMat.max()
        output = outputMat.tolist().index(maxVal)
        print("Machine Predicted: " +str(output))

        def machineImg():
            randVal = random.randint(0,1796)
            target = digits.target[randVal]
            if (target == output):
                imgTrain = digits.images[randVal]
                vfunc = np.vectorize(normalization)
                normalizedImg = vfunc(imgTrain)
                global machineArr
                machineArr = 255-normalizedImg
                # print("MachineArr:\n" + str(machineArr))
                return;
            machineImg()

        machineImg()
        # print("User Img:\n" + str(arr))
        global machineArr
        # print("Machine Predicted:\n" + str(machineArr))
        plt.subplot(1,2,1);
        plt.imshow(arr, cmap=plt.cm.gray, interpolation='nearest')
        plt.subplot(1,2,2);
        plt.imshow(machineArr, cmap=plt.cm.gray, interpolation='nearest')
        plt.show()

    def paint(event):
        # python_green = "#476042"
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        # cv.create_oval(x1, y1, x2, y2, fill="black",width=10)
        cv.create_line(x1, y1, x2, y2, fill="black",width=7)

        draw.line([x1, y1, x2, y2],fill="black",width=7)
        # draw.ellipse([x1, y1, x2*1.2, y2*1.2],fill="black",width=10)

    def clear():
        cv.delete("all")
        draw.rectangle([0, 0, 400, 400],fill="white",width=10)

    root = Tk()
    root.title( "Drawing_float" )

    # Tkinter create a canvas to draw on
    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack(padx=10, pady=10)
    # cv.pack()

    # PIL create an empty image and draw object to draw on memory only, not visible
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)

    # Tkinter canvas drawings (visible)
    # cv.create_line([0, center, width, center], fill='green')
    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)
    buttonPredict=Button(root,text="save",command=save)
    buttonClear=Button(root,text="clear",command=clear)
    buttonPredict.pack()
    buttonClear.pack()
    root.mainloop()

# showPic(6)
training()
testing()
drawingCanvas()
