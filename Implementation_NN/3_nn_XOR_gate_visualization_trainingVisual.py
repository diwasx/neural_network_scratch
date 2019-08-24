#!/bin/python3

import sys
import random
import numpy as np
sys.path.append('..')
from neural_network import NeuralNetwork
import time
from tkinter import *

tk = Tk()
widthSize = 500
heightSize = 500
frameRate = 60
frameSpeed = int(1 / frameRate * 1000)
canvas = Canvas(tk, width=widthSize, height=heightSize, background="black")
tk.title("Drawing_float")
canvas.pack()
inputLen = 2
hiddenLen = 18
outputLen = 1
learningRate = 0.1
n = NeuralNetwork(inputLen, hiddenLen, outputLen)

# With this structure, answer may not be predicted sometimes
# n = NeuralNetwork(2, 2, 1)

training_data = {
    1: {'inputs': np.array([[0],[0]]), 'targets': np.array([[0]])},
    2: {'inputs': np.array([[0],[1]]), 'targets': np.array([[1]])},
    3: {'inputs': np.array([[1],[0]]), 'targets': np.array([[1]])},
    4: {'inputs': np.array([[1],[1]]), 'targets': np.array([[0]])},
    }

while True:

    # Supervised Training with Visualization
    loopForSpeedUp = 250
    for j in range (loopForSpeedUp):
        x = random.choice(list(training_data.values()))
        inputs =  x.get('inputs')
        targets =  x.get('targets')
        n.trainSVLearing(inputs,targets,learningRate)

    x = random.choice(list(training_data.values()))
    inputs =  x.get('inputs')
    targets =  x.get('targets')
    tkSV = n.trainSVLearingVisualization(inputs,targets,learningRate)

    resolution = 10
    cols = widthSize/resolution
    rows = heightSize/resolution
    for i in range(int(cols)):
        for j in range(int(rows)):
            x1 = i/cols
            x2 = j/rows
            inputs = np.array([[x1],[x2]])
            print ("Inputs = "+ str(inputs))
            y = n.feedForward(inputs)
            print ("Output = "+ str(y))
            output = y

            # print("Value of y is = " + str(output))
            color = int(output * 255)
            # print("Value of color is = " + str(color))
            hexColor = format(color, '02x')
            # print("Value of hex is = " + str(hexColor))
            finalColor = "#" + hexColor + hexColor + hexColor
            print("finalColor = " + str(finalColor))

            # rect = canvas.create_rectangle(i*resolution, j*resolution, (i+1)*resolution, (j+1)*resolution, outline='red')
            rect = canvas.create_rectangle(i*resolution, j*resolution, (i+1)*resolution, (j+1)*resolution)
            # canvas.itemconfig(rect, fill="#ff00ff")
            canvas.itemconfig(rect, fill=finalColor)

    tk.after(frameSpeed, tk.update()) # for every give time updates frame 

tk.mainloop()
