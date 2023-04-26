from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rts.player.player import Player

import pip

try:
    import pygame
except ImportError:
    pip.main(['install', '--user', 'pygame'])
    print('import pygame')
    import pygame

class Board:

    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(size = 24)
        self.screen = pygame.display.set_mode((100, 100))        
        self.cell_dimension = 0
        self.padding = 0
        self.width = 0
        self.height = 0
        self.rowNb = -1
        self.columnNb = -1
        self.rowCoord = []
        self.columnCoord = []

    def resize(self, dimension_cell, dimension_map):
        self.cell_dimension = dimension_cell
        self.padding = 20
        self.width = dimension_map[1] * dimension_cell + 2 * self.padding
        self.height = dimension_map[0] * dimension_cell + 2 * self.padding
        self.screen = pygame.display.set_mode((self.width, self.height))        
        self.rowNb = dimension_map[0]
        self.columnNb = dimension_map[1]
        self.rowCoord = [self.padding + i * (self.height - self.padding * 2) / self.rowNb for i in range(self.rowNb + 1)]
        self.columnCoord = [self.padding + i * (self.width - self.padding * 2) / self.columnNb for i in range(self.columnNb + 1)]

    def close(self):
        pygame.quit()

    def refresh(self, terrainMap, units, players: list[Player]):
        pygame.event.pump()

        self.screen.fill("white")
        for i in range(self.rowNb + 1):
            pygame.draw.line(self.screen, "black", (self.padding, self.rowCoord[i]), (self.width - self.padding, self.rowCoord[i]))
        for i in range(self.columnNb + 1):
            pygame.draw.line(self.screen, "black", (self.columnCoord[i], self.padding), (self.columnCoord[i], self.height - self.padding))
        
        self.drawTerrain(terrainMap)

        for unit in units:
            self.drawUnit(unit, players)

        pygame.display.flip()

    def drawTerrain(self, terrainMap):
        for x in range(self.rowNb):
            for y in range(self.columnNb):
                if terrainMap[y * self.rowNb + x] == '1':
                    rect = pygame.Rect(self.padding + x * self.cell_dimension + 1, self.padding + y * self.cell_dimension + 1, self.cell_dimension - 1, self.cell_dimension - 1)
                    pygame.draw.rect(self.screen, (0, 84, 0), rect)

    def drawUnit(self, unit, players: list[Player]):
        borderColor = 'red'
        if unit.player == 0:
            borderColor = 'blue'

        if unit.type.name == "Worker":
            center = (self.padding + (unit.x + 0.5) * self.cell_dimension, self.padding + (unit.y + 0.5) * self.cell_dimension)
            pygame.draw.circle(self.screen, (128, 128, 128), center, self.cell_dimension / 3)
            pygame.draw.circle(self.screen, borderColor, center, self.cell_dimension / 3, 2)
        elif unit.type.name == "Light":
            center = (self.padding + (unit.x + 0.5) * self.cell_dimension, self.padding + (unit.y + 0.5) * self.cell_dimension)
            pygame.draw.circle(self.screen, (213, 94, 0), center, self.cell_dimension / 2.5)
            pygame.draw.circle(self.screen, borderColor, center, self.cell_dimension / 2.5, 2)
        elif unit.type.name == "Heavy":
            center = (self.padding + (unit.x + 0.5) * self.cell_dimension, self.padding + (unit.y + 0.5) * self.cell_dimension)
            pygame.draw.circle(self.screen, (240, 228, 66), center, self.cell_dimension / 2)
            pygame.draw.circle(self.screen, borderColor, center, self.cell_dimension / 2, 2)
        elif unit.type.name == "Ranged":
            center = (self.padding + (unit.x + 0.5) * self.cell_dimension, self.padding + (unit.y + 0.5) * self.cell_dimension)
            pygame.draw.circle(self.screen, (0, 114, 178), center, self.cell_dimension / 2.5)
            pygame.draw.circle(self.screen, borderColor, center, self.cell_dimension / 2.5, 2)
        elif unit.type.name == "Base":
            rect = pygame.Rect(self.padding + unit.x * self.cell_dimension + 1, self.padding + unit.y * self.cell_dimension + 1, self.cell_dimension - 1, self.cell_dimension - 1)
            pygame.draw.rect(self.screen, 'white', rect)
            pygame.draw.rect(self.screen, borderColor, rect, 2)
            for player in players:
                if player.id == unit.player:
                    text = self.font.render(str(player.resources), True, 'black')
                    textCenter = text.get_rect().center
                    rectCenter = rect.center
                    self.screen.blit(text, [rectCenter[i] - textCenter[i] for i in range(len(rect.center))])
        elif unit.type.name == "Barracks":
            rect = pygame.Rect(self.padding + unit.x * self.cell_dimension + 1, self.padding + unit.y * self.cell_dimension + 1, self.cell_dimension - 1, self.cell_dimension - 1)
            pygame.draw.rect(self.screen, (192, 192, 192), rect)
            pygame.draw.rect(self.screen, borderColor, rect, 2)
        elif unit.type.name == "Resource":
            rect = pygame.Rect(self.padding + unit.x * self.cell_dimension + 1, self.padding + unit.y * self.cell_dimension + 1, self.cell_dimension - 1, self.cell_dimension - 1)
            pygame.draw.rect(self.screen, (194, 223, 174), rect)
            text = self.font.render(str(unit.resources), True, 'black')
            textCenter = text.get_rect().center
            rectCenter = rect.center
            self.screen.blit(text, [rectCenter[i] - textCenter[i] for i in range(len(rect.center))])