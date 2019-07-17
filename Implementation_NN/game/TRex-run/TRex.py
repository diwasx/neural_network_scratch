#!/bin/python
import pygame
import random
import time
pygame.init()

highScore = 0
class Dino():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.isJump = False
        self.jumpCount = 15

    def draw(self, win):
        a = pygame.draw.rect(win, (0,255,0), (self.x, self.y, self.width, self.height))
        return a 

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
        global score
        global highScore
        global font
        speed +=.0008
        score+=1
        if (highScore < score):
            highScore = score
        # print(speed)
        gen +=1
        # Clearing window
        win.fill((0,0,0))

        # Scrores
        font = pygame.font.SysFont('Arial', 25)
        win.blit(font.render("Score: "+str(score), True, (255,0,255)), (1000, 50))
        win.blit(font.render("Highscore: "+str(highScore), True, (0,255,255)), (780, 50))

        # Dino part
        rectDino = d1.draw(win)
        d1.width,d1.height = 30,90
        if not(d1.isJump):
            d1.y=320

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
                    c = Obstacle(screenWidth,320,30,30)
                # Bird low
                else:
                    c = Obstacle(screenWidth,350,30,30)
                gen = 0
                obstacles.append(c)

        # Looping thought all obstacle and drawing
        for obstacle in obstacles:
            rectObs = obstacle.draw(win,speed)
            # Deleting obstacle if goes beyond left wall
            if obstacle.x < 0:
                obstacles.pop(obstacles.index(obstacle))

            # Collision detection
            if(rectDino.colliderect(rectObs)):
                # print ("collision")
                global run
                run = False
                # pygame.quit()
        # print(cactii)
        pygame.display.update()
   
def main():
    global screenWidth, screenHeight, win, fps, clock, d1, obstacles, gen, speed, run, score, highScore
    score = 0
    screenWidth, screenHeight = 1200, 400
    win = pygame.display.set_mode((screenWidth,screenHeight))
    pygame.display.set_caption("TRex Run")
    fps = 60
    clock = pygame.time.Clock()
    d1 = Dino(30,320,30,90)
    obstacles = []
    gen = 0
    speed = 1

    # Mainloop
    run = True
    while run:
        clock.tick(fps)
        # pygame.time.delay(int(1/fps*1000))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_DOWN] and d1.x > 0:
            if not(d1.isJump):
                d1.width,d1.height = 90,30
                d1.y +=50

        if not(d1.isJump):
            if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
                d1.isJump = True

        if not(d1.isJump):
            if keys[pygame.K_ESCAPE]: 
                run = False

        else:
            if d1.jumpCount >= -15:
                d1.y -= (d1.jumpCount * 2)
                d1.jumpCount -= 1

            else:
                d1.isJump = False
                d1.jumpCount = 15
        redrawGameWindow()

    win.fill((0,0,0))
    win.blit(font.render("Score: "+str(score), True, (255,0,255)), (1000, 50))
    win.blit(font.render("Highscore: "+str(highScore), True, (0,255,255)), (780, 50))
    win.blit(font.render("Game Over, press Space to play again", True, (0,255,255)), (400, 200))
    time.sleep(1)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            print("Restart")
            main()

        if keys[pygame.K_ESCAPE]:
            print("Quit")
            pygame.quit()

        pygame.display.update()
main()
