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
#Teams are red blue green yellow
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
        self.cards = ["Knight"]*14 + ["Road Building"]*2 + ["Year of Plenty"]*2 + ["Monopoly"]*2 + ["Victory Point"]*5
        shuffle(self.cards)
        print(self.cards)

    def doTurn(self, display, screen):
        roll = 7#self.players[self.turn].rollDice()

        if roll == 7:
            self.players[self.turn].moveRobber()
        else:
            self.resourceProduction(roll)

        self.display(screen)
        display.flip()
        print(roll)

        #Proccess the user action
        
        for i in range(10): #Maximum of 10 actions per turn
            action = self.players[self.turn].decideAction()
            if action == "Build City":
                self.players[self.turn].buildCity()

            elif action == "Build Settlement":
                self.players[self.turn].buildSettlement()

            elif action == "Build Bridge":
                self.players[self.turn].buildBridge()

            elif action == "Buy Development Card":
                self.players[self.turn].buyDevelopmentCard()

            elif action == "Trade":
                self.players[self.turn].tradeRequest()

            elif action == "Use Knight Card":
                self.players[self.turn].useKnightCard()

            elif action == "Use Road Building Card":
                self.players[self.turn].useRoadBuildingCard()

            elif action == "Use Year Of Plenty Card":
                self.players[self.turn].useYearOfPlentyCard()

            elif action == "Use Monopoly Card":
                self.players[self.turn].useMonopolyCard()

            elif action == "Use Victory Point Card":
                self.players[self.turn].useVictoryPointCard()

            else:
                self.endTurn()
                break
            self.display(screen)
            display.flip()
            
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
                    loc = None
                    if vertex.settlement != None:
                        resources[tile.resource] = 1
                        loc = vertex.settlement
                    elif vertex.city != None:
                        resources[tile.resource] = 2
                        loc = vertex.city
                    else:
                        continue
                    self.players[loc].changeResources(resources["Brick"], resources["Lumber"], resources["Ore"], resources["Grain"], resources["Wool"])

    def endGame(self, winner):
        if winner == -1:
            print("Invalid winner")
        else:
            print(winner)
        del self

    def placeCity(self, team, location):
        #Check if settlement of same colour is there
        if self.board.vertices[location].settlement == team:
            if not self.players[team].changeResources(0, 0, -3, -2, 0):
                print("Not enough resources for a city")
                return False
            self.board.vertices[location].placeCity(team)
            print("City has been built")
            return True
        print("Invalid Location")
        return False

    def placeSettlement(self, team, location):
        #Check if empty and same colour bridge is connected, and no neighbouring settlements/cities at all
        if self.board.vertices[location].settlement != None:
            print("Spot is already taken")
            return False
        for vertex in self.board.vertices[location].adjacentNodes:
            if vertex.settlement != None or vertex.city != None:
                print("Too close to a neighbouring building")
                return False
        flag = False
        for bridge in self.board.bridges:
            if self.board.vertices[location] in [bridge.v1, bridge.v2]:
                if bridge.bridge == team:
                    flag = True
        if not flag:
            print("No nearby bridge")
            return False
        if not self.players[team].changeResources(-1, -1, 0, -1, -1):
            print("Not enough resources for a settlement")
            return False
        self.board.vertices[location].placeSettlement(team)
        return True
            
        

    def placeBridge(self, team, location1, location2):
        curBridge = None
        for bridge in self.board.vertices[location1].connectedBridges:
            if self.board.vertices[location2] in [bridge.v1, bridge.v2]:
                curBridge = bridge
        if curBridge == None:
            print("That bridge does not exist")
            return False
        if curBridge.bridge != None:
            print("A bridge already exists there")
            return False
        if curBridge.v1.city == team or curBridge.v2.city == team or curBridge.v1.settlement == team or curBridge.v2.settlement == team:
            if not self.players[team].changeResources(-1, -1, 0, 0, 0):
                print("Not enough resources for a bridge")
                return False
            print("Bridge has been built")
            curBridge.bridge = team
            return True
        for bridge in curBridge.v1.connectedBridges + curBridge.v2.connectedBridges:
            if bridge != curBridge and bridge.bridge == team:
                if not self.players[team].changeResources(-1, -1, 0, 0, 0):
                    print("Not enough resources for a bridge")
                    return False
                print("Bridge has been built")
                curBridge.bridge = team
                return True
        print("Can't build a bridge there")
        return False

    def moveRobber(self, team, location):
        #Check if robber location is valid
        if 0 <= location <= 19:
            self.board.tiles[self.board.currentRobber].setRobber(False)
            self.board.tiles[location].setRobber(True)
            self.board.currentRobber = location
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

        self.vertices[32].settlement = 0
        self.vertices[22].settlement = 0
        self.vertices[18].settlement = 1
        self.vertices[8].city = 1
        self.vertices[10].settlement = 2
        self.vertices[24].settlement = 2
        self.vertices[39].settlement = 3
        self.vertices[16].settlement = 3

        self.bridges = []
        for i in range(len(self.vertices)):
            for j in range(i):
                vertex1 = self.vertices[i]
                vertex2 = self.vertices[j]
                dist = ((vertex2.x - vertex1.x)**2 + (vertex2.y - vertex1.y)**2)**0.5
                if abs(dist - 50) < 1:
                    newBridge = Bridge(vertex1, vertex2)
                    if j == 24 and i == 25:
                        newBridge.bridge = 2
                    elif j == 22 and i == 23:
                        newBridge.bridge = 0
                    elif j == 31 and i == 32:
                        newBridge.bridge = 0
                    elif j == 15 and i == 18:
                        newBridge.bridge = 1
                    elif j == 8 and i == 9:
                        newBridge.bridge = 1
                    elif j ==10 and i == 37:
                        newBridge.bridge = 2
                    elif j == 16 and i == 17:
                        newBridge.bridge = 3
                    elif j == 30 and i == 39:
                        newBridge.bridge = 3
                    self.bridges.append(newBridge)
                    
                    vertex1.adjacentNodes.append(vertex2)
                    vertex2.adjacentNodes.append(vertex1)
                    vertex1.connectedBridges.append(newBridge)
                    vertex2.connectedBridges.append(newBridge)
                

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

        for bridge in self.bridges:
            bridge.update(screen)
    
        for vertex in self.vertices:
            vertex.update(screen)






class Hex:
    radius = 50
    boardCenter = Board.center
    
    def __init__(self, distance, real, imaginary, resource, dice):
        self.distance = distance
        self.real = real
        self.imaginary = imaginary
        self.resource = resource
        self.dice = dice
        self.hasRobber = False
        if self.resource == "Null":
            self.hasRobber = True
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
            print(self.real, self.distance)
            draw.circle(screen, (0, 0, 0), (self.x, self.y), 15, 0)

    def setRobber(self, val):
        self.hasRobber = val



class Player:
    def __init__(self, ID, isAI, game):
        self.resources = {"Brick":4, "Lumber":4, "Ore":4, "Grain":4, "Wool":4}
        self.victory = 0
        self.ID = ID
        self.game = game
        self.cards = []
        self.isAI = isAI
        self.invalidOptions = [False for i in range(11)] #Keeps track of which options dont work for the AI

    def addVictory(self):
        self.victory += 1
        if self.victory >= 10:
            self.game.endGame(self.ID)

    def rollDice(self):
        dice1 = randint(1, 6)
        dice2 = randint(1, 6)
        roll = dice1 + dice2
        return roll

    def changeResources(self, brick, lumber, ore, grain, wool):
        if not self.checkCost(-brick, -lumber, -ore, -grain, -wool):
            return False
        self.resources["Brick"] += brick
        self.resources["Lumber"] += lumber
        self.resources["Ore"] += ore
        self.resources["Grain"] += grain
        self.resources["Wool"] += wool
        return True

    def checkCost(self, brick, lumber, ore, grain, wool):
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

        
        if self.isAI:
            #For node in output, see if we can build a city there
            #If so, return true
            #Else, try next city
            return False
        else:
            inp = input("Enter where you would like to place the city: ")
            try:
                inp = int(inp)
            except:
                return False
            return self.game.placeCity(self.ID, inp)
            

    def buildSettlement(self):

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

        if self.isAI:
            return False
        else:
            inp = input("Enter where you would like to place the bridge (For example: 1 2): ")
            try:
                a, b = inp.split(" ")
                a = int(a)
                b = int(b)
            except:
                return False
            return self.game.placeBridge(self.ID, a, b)

    def buyDevelopmentCard(self):
        if len(self.game.cards) == 0:
            print("There are no more cards to buy")
            return False
        if not changeResources(0, 0, -1, -1, -1):
            print("Can't afford a card")
            return False
        self.cards.append(self.game.cards.pop())
        return True


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
        if "Victory Point" in self.cards:
            self.cards.remove("Victory Point")
        else:
            return False
        self.addVictory()
        return True


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
        self.connectedBridges = []
        self.settlement = None #the team here, None if no team
        self.city = None




    def placeSettlement(self, team):
        self.settlement = team

    def placeCity(self, team):
        self.settlement = None
        self.city = team

    def update(self, screen):

        if self.settlement != None:
            draw.circle(screen, teamColours[self.settlement], (self.x, self.y), 10, 0)
            draw.circle(screen, (0, 0, 0), (self.x, self.y), 10, 1)
        elif self.city != None:
            draw.rect(screen, teamColours[self.city], (self.x-10, self.y-10, 20, 20), 0)
            draw.rect(screen, (0, 0, 0), (self.x-10, self.y-10, 20, 20), 1)

class Bridge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.bridge = None #Which team here, None if empty

    def placeBridge(self, team):
        self.bridge = team

    def update(self, screen):
        if self.bridge == None:
            draw.line(screen, (0, 0, 0), (self.v1.x, self.v1.y), (self.v2.x, self.v2.y), 2)
        else:
            draw.line(screen, (0, 0, 0), (self.v1.x, self.v1.y), (self.v2.x, self.v2.y), 10)
            draw.line(screen, teamColours[self.bridge], (self.v1.x, self.v1.y), (self.v2.x, self.v2.y), 8)
    


