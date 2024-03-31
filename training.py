from simulation import *
from random import shuffle

gamesPerTrial = 25

def train(n):
    for gen in range(n):
        winners = []
        for i in range(gamesPerTrial):
            running = True
            g = Game(True, True, True, True, False, i)

            while running:
                
                g.doTurn(display, None)
                if g.gameOver:
                    running = False
                    winner = g.turn
                if g.totalTurns > 1000:
                    running = False
                    winner = -1

            winningPlayer = None
            
            if winner == -1:
                winningPlayer = Player(1, True, 1, -1)
            else:
                winningPlayer = g.players[winner]

            winners.append(winningPlayer)
        print(gen, n)
        decideActionNNs = []
        moveRobberNNs = []
        vertexValueNNs = []
        bridgeValueNNs = []
        cardValueNNs = []
        for i in range(gamesPerTrial):
            decideActionNNs.append(winners[i].decideAction_NN)
            moveRobberNNs.append(winners[i].moveRobber_NN)
            vertexValueNNs.append(winners[i].vertexValue_NN)
            bridgeValueNNs.append(winners[i].bridgeValue_NN)
            cardValueNNs.append(winners[i].cardValue_NN)
        shuffle(decideActionNNs)
        shuffle(moveRobberNNs)
        shuffle(vertexValueNNs)
        shuffle(bridgeValueNNs)
        shuffle(cardValueNNs)

        writingVals = [i for i in range(gamesPerTrial*4-2)]
        shuffle(writingVals)
        curIndex = 0
        for i in range(gamesPerTrial):
            decideActionNNs[i].writeWeights("CurrentGeneration/DecisionWeights/" + str(writingVals[curIndex]) + ".txt") 
            moveRobberNNs[i].writeWeights("CurrentGeneration/RobberPlacementWeights/" + str(writingVals[curIndex]) + ".txt") 
            vertexValueNNs[i].writeWeights("CurrentGeneration/VertexValuationWeights/" + str(writingVals[curIndex]) + ".txt") 
            bridgeValueNNs[i].writeWeights("CurrentGeneration/BridgeValuationWeights/" + str(writingVals[curIndex]) + ".txt") 
            cardValueNNs[i].writeWeights("CurrentGeneration/CardValuationWeights/" + str(writingVals[curIndex]) + ".txt")
            curIndex += 1
            for j in range(3):
                if i != 0:
                    decideActionNNs[i].mutate()
                    moveRobberNNs[i].mutate()
                    vertexValueNNs[i].mutate()
                    bridgeValueNNs[i].mutate()
                    cardValueNNs[i].mutate()
                    decideActionNNs[i].writeWeights("CurrentGeneration/DecisionWeights/" + str(writingVals[curIndex]) + ".txt") 
                    moveRobberNNs[i].writeWeights("CurrentGeneration/RobberPlacementWeights/" + str(writingVals[curIndex]) + ".txt") 
                    vertexValueNNs[i].writeWeights("CurrentGeneration/VertexValuationWeights/" + str(writingVals[curIndex]) + ".txt") 
                    bridgeValueNNs[i].writeWeights("CurrentGeneration/BridgeValuationWeights/" + str(writingVals[curIndex]) + ".txt") 
                    cardValueNNs[i].writeWeights("CurrentGeneration/CardValuationWeights/" + str(writingVals[curIndex]) + ".txt")
                    curIndex += 1
        for i in range(3):
            newData = Player(1, True, 1, -1)
            newData.decideAction_NN.writeWeights("CurrentGeneration/DecisionWeights/" + str(curIndex) + ".txt")
            newData.moveRobber_NN.writeWeights("CurrentGeneration/RobberPlacementWeights/" + str(curIndex) + ".txt") 
            newData.vertexValue_NN.writeWeights("CurrentGeneration/VertexValuationWeights/" + str(curIndex) + ".txt") 
            newData.bridgeValue_NN.writeWeights("CurrentGeneration/BridgeValuationWeights/" + str(curIndex) + ".txt") 
            newData.cardValue_NN.writeWeights("CurrentGeneration/CardValuationWeights/" + str(curIndex) + ".txt")
            curIndex += 1


train(900)

#quit()
                







        
