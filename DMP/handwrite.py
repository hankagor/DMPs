import pygame
import numpy as np
import sys


def handwrite(letters):
    for i in range (0, len(letters)):
        c = letters[i]

        pygame.init()
        width, height = 800, 600
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Draw the letter " + c)
        drawing = False
        color = (255,255,255)

        points = []
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    drawing = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    running = False
                elif event.type == pygame.MOUSEMOTION and drawing:
                    points.append(event.pos)
                    pygame.draw.circle(screen, color, event.pos, 5)

            pygame.display.flip()
        print(len(points))

        with open("parabola.txt", 'w') as f:
            for point in points:
                f.write(f"{point[0]:.2f},{point[1]:.2f}\n")
                
        pygame.quit()



if __name__ == "__main__":
    string = sys.argv[1]
    print(string)
    handwrite(string)