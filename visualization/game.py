import pygame
import numpy as np
 
screen = pygame.display.set_mode((1400, 700))
running = 1

states = "data/save_runs/states/sarsa_tabular__horsetrack__0.0__0.5__0.0__0.01__False__0.0.dat"
q_values = "data/save_runs/q_values/sarsa_tabular__horsetrack__0.0__0.5__0.0__0.01__False__0.0.dat"

state_data = []
with open(states, 'r') as f:
    for line in f.readlines():
        state_data = [float(x.replace("\n", "")) for x in line.split(", ")]

q_data = []
with open(q_values, 'r') as f:
    for line in f.readlines():
        q_data = [x for x in line.split(", ")]
print(q_data)


interval = 1370 / 50.0
red = (255, 0, 0)
green = (0, 255, 0)
black = (0, 0, 0)

row0 = 400
row1 = 200

current_location_index = 0


def draw_pane(x, y, size, text, screen, myfont):
    pygame.draw.rect(screen, black, (x, y, 50, 100), 2)
    screen.blit(myfont.render("1234", 1, black), (50, 10))



while running:
    pygame.font.init()
    myfont = pygame.font.SysFont('Helvetica', 30)
    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        running = 0

    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 0, 0), (20, 400), (1370, 400))

    pygame.draw.circle(screen, red, (20 + int(state_data[current_location_index] * interval), 400), 8)

    next_button = pygame.draw.rect(screen, green, (150,450,100,50))
    pygame.draw.rect(screen, red, (550,450,100,50))

    # draw ticks
    for i in range(51):
        pygame.draw.line(screen, (0, 0, 0), (20 + i*27, 390), (20 + i*27, 410))
        label = myfont.render("{}".format(i), 1, (0, 0, 0))
        screen.blit(label, (15 + i*27, 420))

    draw_pane(10, 10, (10, 10), "1234", screen, myfont)


    for i in range(51):
        pass

    for i in range(51):
        pass

    # if pygame.key.get_pressed()[pygame.K_RIGHT] != 0:
    #     current_location_index += 1

    # mouse = (0, 0)
    # mouse = pygame.mouse.get_pos()
    # # while mouse == (0, 0):
    # #     for event in pygame.event.get():
    # #         if event.type == pygame.MOUSEBUTTONDOWN:
    # #             mouse = pygame.mouse.get_pos()
    # #             print(mouse)


    # if next_button.collidepoint(mouse):
    #     # mouse = (0,0)
    #     current_location_index += 1

    # pygame.draw.aaline(screen, (0, 0, 255), (639, 0), (0, 479))
    pygame.display.flip()