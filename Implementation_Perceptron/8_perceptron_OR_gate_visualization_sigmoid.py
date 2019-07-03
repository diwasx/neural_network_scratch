#!/bin/python3

import sys
sys.path.append('..')
from perceptron import Perceptron
from tkinter import * 
import random
import time

tk = Tk()
widthSize = 500
heightSize = 500
frameRate = 60
frameSpeed = int(1 / frameRate * 1000)

canvas = Canvas(tk, width=widthSize, height=heightSize, background="black")
tk.title("Drawing_float")
canvas.pack()

inputLen = 3
learnRate = 0.1
bias = 1
p = Perceptron(inputLen,"sigmoid")

def classification(x):
    if ( x < 0.5):
        return 0
    else:
        return 1

# for i in range(20):
def training():
    inputs = [bias,0,0]
    target = 0
    p.train(inputs,target, learnRate, classification)
    inputs = [bias,0,1]
    target = 1
    p.train(inputs,target, learnRate, classification)
    inputs = [bias,1,0]
    target = 1
    p.train(inputs,target, learnRate, classification)
    inputs = [bias,1,1]
    target = 1
    p.train(inputs,target, learnRate, classification)

# for i in range(100):
while True:
    training()

    resolution = 10
    cols = widthSize/resolution
    rows = heightSize/resolution
    for i in range(int(cols)):
        for j in range(int(rows)):
            x1 = i/cols
            x2 = j/rows
            inputs = [bias,x1,x2]
            y = p.guess(inputs)
            output = classification(y)
            # output = y

            # print("Value of y is = " + str(output))
            # color = int(output * 255)
            # print("Value of color is = " + str(color))
            # hexColor = format(color, '02x')
            # print("Value of hex is = " + str(hexColor))
            # finalColor = "#" + hexColor + hexColor + hexColor
            # print("finalColor = " + str(finalColor))

            if (output == 0):
                color = '#000000000'
            else:
                color = '#fffffffff'
            
            rect = canvas.create_rectangle(i*resolution, j*resolution, (i+1)*resolution, (j+1)*resolution, outline='red')
            # rect = canvas.create_rectangle(i*resolution, j*resolution, (i+1)*resolution, (j+1)*resolution)
            # canvas.itemconfig(rect, fill="#ff00ff")
            # canvas.itemconfig(rect, fill=finalColor)
            canvas.itemconfig(rect, fill=color)

    tk.after(frameSpeed, tk.update()) # for every give time updates frame 

tk.mainloop()
