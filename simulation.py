from pygame import *
from random import *
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
    def __init__(self, p0, p1, p2, p3):
        #pn True is AI, False is human player
        self.board = Board()
        self.players = [Player(0, p0, self), Player(1, p1, self), Player(2, p2, self), Player(3, p3, self)]
        self.turn = 0

    def doTurn(self):
        action = self.players[self.turn].decideAction()
        if action == "Build City":
            self.players[self.turn].buildCity()
        elif action == "Build Settlement":
            self.players[self.turn].buildSettlement()
        else:
            self.endTurn()
        for player in self.players:
            if player.victory >= 10:
                self.endGame(player)
        self.doTurn()

    def endTurn(self):
        self.turn += 1

    def endGame(self, winner):
        if winner == -1:
            print("Invalid winner")
        else:
            print(winner)
        del self

    def display(self, screen):
        self.board.display(screen)



class Board:
    center = (WIDTH/2, HEIGHT/2)
        
    def __init__(self):
        r2b2 = 3**0.5/2
        self.tiles = [Hex(0, 0, 0, "Null"),
                      Hex(2, 1, 0, "Lumber"), Hex(2, -1, 0, "Lumber"), Hex(2, 0.5, r2b2, "Grain"), Hex(2, -0.5, r2b2, "Ore"), Hex(2, 0.5, -r2b2, "Wool"), Hex(2, -0.5, -r2b2, "Brick"),
                      Hex(3/r2b2, 0, 1, "Grain"), Hex(3/r2b2, 0, -1, "Wool"), Hex(3/r2b2, r2b2, 0.5, "Wool"), Hex(3/r2b2, r2b2, -0.5, "Brick"), Hex(3/r2b2, -r2b2, 0.5, "Lumber"), Hex(3/r2b2, -r2b2, -0.5, "Grain"),
                      Hex(4, 1, 0, "Ore"), Hex(4, -1, 0, "Grain"), Hex(4, 0.5, r2b2, "Wool"), Hex(4, -0.5, r2b2, "Brick"), Hex(4, 0.5, -r2b2, "Lumber"), Hex(4, -0.5, -r2b2, "Ore")]
        
        self.vertices = []

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
        self.hasRobber = self.resource == "Null"
        colours = {"Brick":(255, 0, 0), "Lumber":(150, 75, 0), "Ore": (150, 150, 150), "Grain": (255, 255, 0), "Wool": (255, 255, 255), "Null":(100, 100, 30)}
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
        (self.x - Hex.radius * 3 ** 0.5 / 2, self.y - Hex.radius / 2)], 0)
        if self.hasRobber:
            draw.circle(screen, (0, 0, 0), (self.x, self.y), 15, 0)



class Player:
    def __init__(self, ID, isAI, game):
        self.resources = {"Brick":0, "Lumber":0, "Ore":0, "Grain":0, "Wool":0}
        self.victory = 0
        self.ID = ID
        self.game = game
        self.cards = []
        self.isAI = isAI

    def addVictory(self):
        self.victory += 1
        if self.victory >= 10:
            self.game.endGame(self.ID)

    def rollDice(self):
        dice1 = randint(1, 6)
        dice2 = randint(1, 6)
        roll = dice1 + dice2
        return roll

    def changeResources(brick, lumber, ore, grain, wool):
        self.resources["Brick"] += brick
        self.resources["Lumber"] += lumber
        self.resources["Ore"] += ore
        self.resources["Grain"] += grain
        self.reosurces["Wool"] += wool

    def moveRobber(self):
        if self.isAI:
            pass
        else:
            inp = input("Enter the tile to move the robber to: ")
            return int(inp)

    def decideAction(self):
        actions = {"1":"Build City", "2":"Build Settlement"}
        if self.isAI:
            pass
        else:
            print("1: Build City")
            inp = input("Enter the action you would like to take: ")
            return actions[int(inp)]

    def buildCity(self):
        if self.isAI:
            pass
        else:
            pass

    def buildSettlement(self):
        if self.isAI:
            pass
        else:
            pass

    def buildBridge(self):
        if self.isAI:
            pass
        else:
            pass

    def tradeRequest(self):
        if self.isAI:
            pass
        else:
            pass




"""
Turn:
roll dice and get production
built settlements
build cities
buy development card
play development cards (cant be one you just bought cept for victory cards)
"""

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

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bridge1 = None
        self.bridge2 = None #Bridge is labelled with 0, 1, 2, 3 depending on player
        self.bridge3 = None #always None if it is an outer vertex
        self.town = None #the town here
        self.city = None


