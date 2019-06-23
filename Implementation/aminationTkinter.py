#!/bin/python3

from tkinter import * 
import random
import time

tk = Tk()
widthSize = 800
heightSize = 600
frameRate = 60
# frameRate = 240
frameSpeed = int(1 / frameRate * 1000)

canvas = Canvas(tk, width=widthSize, height=heightSize)
tk.title("Drawing")
canvas.pack()

ball = canvas.create_oval(10, 10, 60, 60, fill="yellow")
xspeed = 20
yspeed = 10

# for i in range(100):
while True:
    # canvas.move(ball, 1, 0)
    canvas.move(ball, xspeed, yspeed)
    pos = canvas.coords(ball)
    # pos [ left, top, right, bottom ]
    if pos[3] >= heightSize or pos[1] <= 0:
        yspeed = -yspeed
    if pos[2] >= widthSize or pos[0] <= 0:
        xspeed = -xspeed

    # tk.update()
    # time.sleep(0.01)
    # Another method to update
    tk.after(frameSpeed, tk.update()) # for every give time updates frame 

tk.mainloop()
