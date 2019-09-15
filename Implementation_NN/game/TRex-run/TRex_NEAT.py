#!/bin/python
import pygame
import random
import time
import sys
import numpy as np
sys.path.append('../../..')
from neural_network import NeuralNetwork
pygame.init()

highScore = 0
population = 130
# population = 180
generation = 1
class Dino():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.scoreVal = 0
        self.fitness = 0
        self.height = height
        self.isJump = False
        self.jumpCount = 15
        # self.brain = NeuralNetwork(6,100,3)
        # self.brain = NeuralNetwork(6,50,3)
        # self.brain = NeuralNetwork(6,20,3)
        self.brain = NeuralNetwork(6,80,3)
        self.color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

    def draw(self, win):
        self.rectangle = pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))

        # return a 
    
    def collisionDetect(self, rectObs):
        if(self.rectangle.colliderect(rectObs)):
            global dinosours
            # print("Dinosour poped" + str(self))
            savedDino.append(self)
            index = dinosours.index(self)
            dinosours.pop(index)
            print("Dinosours left " + str(len(dinosours)))

    def think(self):
        global obstacles, speed
        if (len(obstacles) != 0):
            aiSeeX = obstacles[0].x
            aiSeeY = obstacles[0].y
            aiSeeWidth = obstacles[0].width
            aiSeeHeight = obstacles[0].height
            aiSeeSpeed = speed
            if self.isJump == True:
                aiSeeJump = 1
            else:
                aiSeeJump = 0
            inputs = np.array([[aiSeeX],[aiSeeY], [aiSeeWidth], [aiSeeHeight], [aiSeeSpeed], [aiSeeJump]])
            output = self.brain.feedForward(inputs)
            # print(output)
            maxVal = output.max()
            output = output.tolist().index(maxVal)
            # print(output)
            if (output == 0):
                # print("Duck")
                if not(self.isJump):
                    # Ducking
                    self.width,self.height = 90,30
                    self.y +=60
            elif (output == 1):
                # print("Jump")
                if not(self.isJump):
                        self.isJump = True
            # else: 
            #     print("Do nothing")

        if (self.isJump):
            if self.jumpCount >= -15:
                self.y -= (self.jumpCount * 2)
                self.jumpCount -= 1
            else:
                self.isJump = False
                self.jumpCount = 15

            # child = self.brain.copy()
            # child = brain.mutate(0.01)

class Obstacle():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vel = 5

    def draw(self, win, speed):
        b = pygame.draw.rect(win, (255,0,0), (self.x, self.y, self.width, self.height))
        self.x -= self.vel * speed
        return b

def redrawGameWindow():
        global gen
        global speed
        global obstacles 
        global highScore
        global score
        global font
        global dinosours
        global generation
        speed +=.0008
        score +=1
        if (highScore < score):
            highScore = score
        gen +=1
        # Clearing window
        win.fill((0,0,0))
        # Scrores
        font = pygame.font.SysFont('Arial', 25)
        win.blit(font.render("Score: "+str(score), True, (255,0,255)), (1000, 50))
        win.blit(font.render("Highscore: "+str(highScore), True, (0,255,255)), (780, 50))
        win.blit(font.render("Generation: "+str(generation), True, (255,255,0)), (580, 50))
        pygame.draw.line(win, (255,0,0), (0,400), (screenWidth, 400))

        # Dino part
        for dinosour in dinosours:
            dinosour.draw(win)
            dinosour.scoreVal+=1
            dinosour.width,dinosour.height = 30,90
            if not(dinosour.isJump):
                dinosour.y=310

        # Obstacle part
        rand = random.randint(0,50)
        if (gen > 25):
            if (rand == 0):
                tmp = random.randint(0,15)
                # print (tmp)
                # Cactus normal
                if (tmp > 0 and tmp <=4):
                    c = Obstacle(screenWidth,340,50,60)
                # Cactus large
                elif (tmp > 4 and tmp <=8):
                    c = Obstacle(screenWidth,340,80,60)
                # Cactus tall
                elif (tmp > 8 and tmp <=12):
                    c = Obstacle(screenWidth,310,20,90)
                # Bird high
                elif (tmp == 13):
                    c = Obstacle(screenWidth,270,30,30)
                # Bird medium
                elif (tmp == 14):
                    c = Obstacle(screenWidth,80,30,260)
                # Bird low
                else:
                    c = Obstacle(screenWidth,110,30,260)
                gen = 0
                obstacles.append(c)

        # Looping thought all obstacle and drawing
        for obstacle in obstacles:
            rectObs = obstacle.draw(win,speed)
            # Deleting obstacle if goes beyond left wall
            if obstacle.x < 0:
                obstacles.pop(obstacles.index(obstacle))

            for dinosour in dinosours:
                dinosour.collisionDetect(rectObs)
        pygame.display.update()

dinosours = [None] * population
for i in range(population):
    dinosours[i] = Dino(30,310,30,90)
   
def main():
    global screenWidth, screenHeight, win, fps, clock, dinosours, obstacles, gen, speed, run, score, highScore, savedDino
    savedDino = []
    score = 0
    screenWidth, screenHeight = 1200, 500
    win = pygame.display.set_mode((screenWidth,screenHeight))
    pygame.display.set_caption("TRex Run")
    fps = 60
    # fps = 120
    clock = pygame.time.Clock()
    # dinosour = Dino(30,310,30,90)
    obstacles = []
    gen = 0
    speed = 1

    # Mainloop
    run = True
    while run:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
        for dinosour in dinosours:
            dinosour.think()
        redrawGameWindow()


        # If all dino dies, loop end
        if len(dinosours) == 0:
            run = False

    win.fill((0,0,0))
    time.sleep(0.2)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            pygame.quit()
    # print("Restart")
    nextGen()

def nextGen():
    global generation
    global savedDino
    global total
    global population
    # print ("\n" + str(savedDino))

    # Calculate fitness value
    total = 0
    for dino in savedDino:
        total += dino.scoreVal
    # print ("Total = "+ str(total))
    maxFitness = 0
    for dino in savedDino:
        dino.fitness = dino.scoreVal / total
        # print (str(dino.fitness))
        if maxFitness < dino.fitness:
            maxFitness = dino.fitness

    # Selection based on fitness value
    for i in range(population):
        index = 0 
        r = random.uniform(0,1);
        while (r>0):
            r = r - savedDino[index].fitness;
            index +=1;
        index -=1
        tmp = savedDino[index]
        child = Dino(30,310,30,90)
        # child.brain = tmp.brain.copy()
        child.brain.weightsHid = tmp.brain.weightsHid
        child.brain.weightsOut = tmp.brain.weightsOut
        child.brain.biasHid = tmp.brain.biasHid
        child.brain.biasOut = tmp.brain.biasOut
        child.brain.mutate(0.03)
        dinosours.append(child)
    # print(len(dinosours))
    # print(dinosours)
    generation +=1
    main()

main()
