from pygame import *
"""
Classes:
game (includes board state and the player states, current turn, who is AI and who isn't, delay for the AI turns)

board (includes the hex layout and all the things on the board)

hex (simple coloured hex with a number on it)

player (has their own funcs but can be AI or in person controlled)

cards

bridge, cities, and settlements

dice


Turn:
roll dice and get production
built settlements
build cities
buy development card
play development cards (cant be one you just bought cept for victory cards)

if you rolled a 7, do robber stuff


Check if you won
"""

WIDTH, HEIGHT = 1280, 720

init()

class Game:
    def __init__(self):
        self.board = Board()

    def endGame(self, winner):
        if winner == -1:
            print("Invalid winner")
        else:
            print(winner)

    def display(self, screen):
        self.board.display(screen)



class Board:
    center = (WIDTH/2, HEIGHT/2)
        
    def __init__(self):
        r2b2 = 3**0.5/2
        self.tiles = [Hex(0, 0, 0, "Null"),
                      Hex(2, 1, 0, "Brick"), Hex(2, -1, 0, "Brick"), Hex(2, 0.5, r2b2, "Brick"), Hex(2, -0.5, r2b2, "Brick"), Hex(2, 0.5, -r2b2, "Brick"), Hex(2, -0.5, -r2b2, "Brick"),
                      Hex(3/r2b2, 0, 1, "Brick"), Hex(3/r2b2, 0, -1, "Brick"), Hex(3/r2b2, r2b2, 0.5, "Brick"), Hex(3/r2b2, r2b2, -0.5, "Brick"), Hex(3/r2b2, -r2b2, 0.5, "Brick"), Hex(3/r2b2, -r2b2, -0.5, "Brick"),
                      Hex(4, 1, 0, "Brick"), Hex(4, -1, 0, "Brick"), Hex(4, 0.5, r2b2, "Brick"), Hex(4, -0.5, r2b2, "Brick"), Hex(4, 0.5, -r2b2, "Brick"), Hex(4, -0.5, -r2b2, "Brick")]
        

    def display(self, screen):
        for tile in self.tiles:
            tile.display(screen)



class Hex:
    radius = 50
    boardCenter = Board.center
    
    def __init__(self, distance, real, imaginary, resource):
        self.distance = distance
        self.real = real
        self.imaginary = imaginary
        self.resource = resource
        colours = {"Brick":(255, 0, 0), "Null":(150, 150, 150)}
        self.col = colours[self.resource]
        self.x = Hex.boardCenter[0] + Hex.radius*distance*real*(3**0.5)/2
        self.y = Hex.boardCenter[1] + Hex.radius*distance*imaginary*(3**0.5)/2

    def display(self, screen):
        draw.polygon(screen, self.col, [
        (self.x, self.y - Hex.radius),
        (self.x + Hex.radius * 3 ** 0.5 / 2, self.y - Hex.radius / 2),
        (self.x + Hex.radius * 3 ** 0.5 / 2, self.y + Hex.radius / 2),
        (self.x, self.y + Hex.radius),
        (self.x - Hex.radius * 3 ** 0.5 / 2, self.y + Hex.radius / 2),
        (self.x - Hex.radius * 3 ** 0.5 / 2, self.y - Hex.radius / 2)], 2)



class Player:
    def __init__(self, ID, game):
        self.resources = {"Brick":0, "Lumber":0, "Ore":0, "Grain":0, "Wool":0}
        self.victory = 0
        self.ID = ID
        self.game = game

    def addVictory(self):
        self.victory += 1
        if self.victory >= 10:
            self.game.endGame(self.ID)



class DevelopmentCard:
    def __init__(self, card):
        pass

    def useCard(self, player):
        if card == "Knight":
            pass
        elif card == "Monopoly":
            pass
        elif card == "Library":
            player.addVictory()
