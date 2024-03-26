from pygame import *
from random import *
from math import cos, sin, pi
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

teamColours = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]


#One full Catan simulation from start until someone wins
class Game:
    def __init__(self, p0, p1, p2, p3, show):
        #pn True is AI, False is human player
        #The ID is which team you are on (team 0, 1, 2, 3)
        self.board = Board()
        self.players = [Player(0, p0, self), Player(1, p1, self), Player(2, p2, self), Player(3, p3, self)]
        self.turn = 0
        self.show = show

    def doTurn(self):
        roll = self.players[self.turn].rollDice()

        if roll == 7:
            self.players[self.turn].moveRobber()
        else:
            self.resourceProduction(roll)

        #Proccess the user action
        action = self.players[self.turn].decideAction()
        for i in range(5): #Maximum of 5 actions per turn
            if action == "Build City":
                if self.players[self.turn].buildCity():
                    break
                else:
                    continue
            elif action == "Build Settlement":
                if self.players[self.turn].buildSettlement():
                    break
                else:
                    continue
            elif action == "Build Bridge":
                if self.players[self.turn].buildBridge():
                    break
                else:
                    continue
            elif action == "Buy Development Card":
                if self.players[self.turn].buyDevelopmentCard():
                    break
                else:
                    continue
            elif action == "Trade":
                if self.players[self.turn].tradeRequest():
                    break
                else:
                    continue
            elif action == "Use Knight Card":
                if self.players[self.turn].useKnightCard():
                    break
                else:
                    continue
            elif action == "Use Road Building Card":
                if self.players[self.turn].useRoadBuildingCard():
                    break
                else:
                    continue
            elif action == "Use Year Of Plenty Card":
                if self.players[self.turn].useYearOfPlentyCard():
                    break
                else:
                    continue
            elif action == "Use Monopoly Card":
                if self.players[self.turn].useMonopolyCard():
                    break
                else:
                    continue
            elif action == "Use Victory Point Card":
                if self.players[self.turn].useVictoryPointCard():
                    break
                else:
                    continue
            else:
                self.endTurn()
            
        for player in self.players:
            if player.victory >= 10:
                self.endGame(player)

    def endTurn(self):
        self.turn = (self.turn+1)%4

    def resourceProduction(self, rollValue):
        for tile in self.board.tiles:
            if tile.dice == rollValue:
                for vertex in tile.vertices:
                    resources = {"Brick":0, "Lumber":0, "Ore":0, "Grain":0, "Wool":0}
                    if vertex.settlement != None:
                        resources[tile.resource] = 1
                    elif vertex.city != None:
                        resources[tile.resource] = 2
                    else:
                        continue
                    self.players[vertex].changeResources(resources["Brick"], resources["Lumber"], resources["Ore"], resources["Grain"], resources["Wool"])

    def endGame(self, winner):
        if winner == -1:
            print("Invalid winner")
        else:
            print(winner)
        del self

    def placeCity(self, team, location):
        #Check if settlement of same colour is there
        if self.board.vertices[location].settlement == team:
            self.board.vertices[location].placeCity(team)
            return True
        return False

    def placeSettlement(self, team, location):
        #Check if empty and same colour bridge is connected, and no neighbouring settlements/cities at all
        if self.board.vertices[location].settlement != None:
           return False
        for vertex in self.board.vertices[location].adjacentNodes:
            if vertex.settlement != None or vertex.city != None:
                return False
        flag = False
        for bridge in self.board.bridges:
            if self.board.vertices[location] in [bridge.v1, bridge.v2]:
                if bridge.bridge == team:
                    flag = True
        self.board.vertices[location].placeSettlement(team)
        return True
            
        

    def placeBridge(self, team, location1, location2):
        #Check if empty and connected to another bridge of same colour
        pass

    def moveRobber(self, team, location):
        #Check if robber location is valid
        if 0 <= location <= 19:
            self.board.tiles[self.board.currentRobber].setRobber(False)
            self.board.tiles[location].setRobber(True)
            options = []
            for vertex in self.board.tiles[location].vertices:
                if vertex.settlement != team and vertex.city != team:
                    
                    options.append(vertex)
                    
                    
            return True
        return False

    def totalResources(self, team):
        return self.players[team].resources["Brick"] + self.players[team].resources["Lumber"] + self.players[team].resources["Ore"] + self.players[team].resources["Wool"] + self.players[team].resources["Grain"]

    def display(self, screen):
        if self.show:
            self.board.display(screen)



class Board:
    center = (WIDTH/2, HEIGHT/2)
        
    def __init__(self):
        r3b2 = 3**0.5/2

        self.vertices = []
        for i in range(6):
            self.vertices.append(Vertex(Board.center[0] + 50*cos(pi*i/3+pi/6), Board.center[1] + 50*sin(pi*i/3 + pi/6)))
        for i in range(6):
            hexCenterX = Board.center[0] + (3)*50*cos(pi*i/3+pi/6)
            hexCenterY = Board.center[1] + (3)*50*sin(pi*i/3+pi/6)
            for j in range(6):
                self.vertices.append(Vertex(hexCenterX + 50*cos(pi*j/3+pi/6), hexCenterY + 50*sin(pi*j/3+pi/6)))
        for i in range(6):
            hexCenterX = Board.center[0] + 4*r3b2*50*cos(i*pi/3)
            hexCenterY = Board.center[1] + 4*r3b2*50*sin(i*pi/3)
            self.vertices.append(Vertex(hexCenterX + 50*cos(i*pi/3+pi/6), hexCenterY + 50*sin(i*pi/3 + pi/6)))
            self.vertices.append(Vertex(hexCenterX + 50*cos(i*pi/3-pi/6), hexCenterY + 50*sin(i*pi/3 - pi/6)))

        self.bridges = []
        for i in range(len(self.vertices)):
            for j in range(i):
                vertex1 = self.vertices[i]
                vertex2 = self.vertices[j]
                dist = ((vertex2.x - vertex1.x)**2 + (vertex2.y - vertex1.y)**2)**0.5
                if abs(dist - 50) < 1:
                    self.bridges.append(Bridge(vertex1, vertex2))
                    vertex1.adjacentNodes.append(vertex2)
                    vertex2.adjacentNodes.append(vertex1)
                

        self.tiles = [Hex(0, 0, 0, "Null", -1),
                      Hex(2, 1, 0, "Lumber", 3), Hex(2, -1, 0, "Lumber", 11), Hex(2, 0.5, r3b2, "Grain", 4), Hex(2, -0.5, r3b2, "Ore", 3), Hex(2, 0.5, -r3b2, "Wool", 4), Hex(2, -0.5, -r3b2, "Brick", 6),
                      Hex(3/r3b2, 0, 1, "Grain", 6), Hex(3/r3b2, 0, -1, "Wool", 2), Hex(3/r3b2, r3b2, 0.5, "Wool", 5), Hex(3/r3b2, r3b2, -0.5, "Brick", 10), Hex(3/r3b2, -r3b2, 0.5, "Lumber", 8), Hex(3/r3b2, -r3b2, -0.5, "Grain", 12),
                      Hex(4, 1, 0, "Ore", 8), Hex(4, -1, 0, "Grain", 9), Hex(4, 0.5, r3b2, "Wool", 11), Hex(4, -0.5, r3b2, "Brick", 5), Hex(4, 0.5, -r3b2, "Lumber", 9), Hex(4, -0.5, -r3b2, "Ore", 10)]

        self.currentRobber = 0

        for tile in self.tiles:
            for vertex in self.vertices:
                dist = ((vertex.x-tile.x)**2 + (vertex.y-tile.y)**2)**0.5
                if abs(dist - 50) < 1:
                    tile.vertices.append(vertex)



            

    def display(self, screen):
        for tile in self.tiles:
            tile.update(screen)
    
        for vertex in self.vertices:
            vertex.update(screen)

        for bridge in self.bridges:
            bridge.update(screen)




class Hex:
    radius = 50
    boardCenter = Board.center
    
    def __init__(self, distance, real, imaginary, resource, dice):
        self.distance = distance
        self.real = real
        self.imaginary = imaginary
        self.resource = resource
        self.dice = dice
        self.hasRobber = self.resource == "Null"
        colours = {"Brick":(255, 0, 0), "Lumber":(150, 75, 0), "Ore": (150, 150, 150), "Grain": (255, 255, 0), "Wool": (255, 255, 255), "Null":(100, 100, 30)}
        self.col = colours[self.resource]
        self.x = Hex.boardCenter[0] + Hex.radius*distance*real*(3**0.5)/2
        self.y = Hex.boardCenter[1] + Hex.radius*distance*imaginary*(3**0.5)/2
        self.vertices = []

    def update(self, screen):
        draw.polygon(screen, self.col, [
        (self.x, self.y - Hex.radius),
        (self.x + Hex.radius * 3 ** 0.5 / 2, self.y - Hex.radius / 2),
        (self.x + Hex.radius * 3 ** 0.5 / 2, self.y + Hex.radius / 2),
        (self.x, self.y + Hex.radius),
        (self.x - Hex.radius * 3 ** 0.5 / 2, self.y + Hex.radius / 2),
        (self.x - Hex.radius * 3 ** 0.5 / 2, self.y - Hex.radius / 2)], 0)
        if self.hasRobber:
            draw.circle(screen, (0, 0, 0), (self.x, self.y), 15, 0)

    def setRobber(self, val):
        self.hasRobber = val



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
        if self.checkCost(brick, lumber, ore, grain, wool):
            return False
        self.resources["Brick"] += brick
        self.resources["Lumber"] += lumber
        self.resources["Ore"] += ore
        self.resources["Grain"] += grain
        self.reosurces["Wool"] += wool
        return True

    def checkCost(brick, lumber, ore, grain, wool):
        return self.resources["Brick"] >= brick and self.resources["Lumber"] >= lumber and self.resources["Ore"] >= ore and self.resources["Grain"] >= grain and self.resources["Wool"] >= wool

    # These functions return true if succesful/possible, otherwise returns false
    def decideAction(self):
        actions = {1:"Build Settlement", 2:"Build City", 3:"Build Bridge", 4:"Buy Development Card", 5:"Trade", 6:"Use Knight Card", 7:"Use Road Building Card", 8:"Use Year of Plenty Card", 9:"Use Monopoly Card", 10:"Use Victory Point Card",  11:"End Turn"}
        if self.isAI:
            pass
    
        else:
            for i in range(1, 12):
                print(i, ":", actions[i])
            while True:
                try:
                    inp = int(input("Enter the action you would like to take: "))
                    if inp < 1 or inp > 11:
                        raise
                    break
                except:
                    pass

            return actions[int(inp)]

    def moveRobber(self):
        if self.isAI:
            pass
        else:
            inp = input("Enter the tile to move the robber to: ")
            try:
                inp = int(inp)
            except:
                return False
            return self.game.moveRobber(self.ID, inp)


    
    def buildCity(self):
        if not changeResources(0, 0, -3, -2, 0):
            return False
        
        if self.isAI:
            #For node in output, see if we can build a city there
            #If so, return true
            return False
        else:
            inp = input("Enter where you would like to place the city: ")
            try:
                inp = int(inp)
            except:
                return False
            return self.game.placeCity(self.ID, inp)
            

    def buildSettlement(self):
        if not changeResources(-1, -1, 0, -1, -1):
            print("Not enough resources for a settlement")
        if self.isAI:
            return False
        else:
            inp = input("Enter where you would like to place the settlement: ")
            try:
                inp = int(inp)
            except:
                return False
            return self.game.placeSettlement(self.ID, inp)
            

    def buildBridge(self):
        if not changeResources(-1, -1, 0, -1, -1):
            print("Not enough resources for a bridge")
        if self.isAI:
            return False
        else:
            inp = input("Enter where you would like to place the bridge (For example: 1 2): ")
            try:
                inp = int(inp)
            except:
                return False
            return self.game.placeBridge(self.ID, inp)

    def buyDevelopmentCard(self):
        if self.isAI:
            pass
        else:
            pass

    def tradeRequest(self):
        if self.isAI:
            pass
        else:
            pass

    def useKnightCard(self):
        if self.isAI:
            pass
        else:
            pass

    def useRoadBuildingCard(self):
        if self.isAI:
            pass
        else:
            pass

    def useYearOfPlentyCard(self):
        if self.isAI:
            pass
        else:
            pass

    def useMonopolyCard(self):
        if self.isAI:
            pass
        else:
            pass

    def useVictoryPointCard(self):
        self.addVictory()


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
        self.adjacentNodes = [] #Which nodes are next to it
        self.settlement = None #the team here, None if no team
        self.city = None

    def update(self, screen):
        draw.circle(screen, (255, 255, 255), (self.x, self.y), 10, 0)
        if self.settlement != None:
            pass


    def placeSettlement(self, team):
        self.settlement = team

    def placeCity(self, team):
        self.settlement = None
        self.city = team

class Bridge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.bridge = None #Which team here, None if empty

    def placeBridge(self, team):
        self.bridge = team

    def update(self, screen):
        draw.line(screen, (255, 255, 255), (self.v1.x, self.v1.y), (self.v2.x, self.v2.y), 3)
    


