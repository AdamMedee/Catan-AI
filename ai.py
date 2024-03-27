import torch
import torch.nn as nn
import torch.nn.functional as F

import random as rnd
from scipy import special

class AI_Agent(nn.Module):
    
    NUM_HIDDEN_LAYERS = 2 # Minimum of 1 hidden layer, otherwise change init
    NUM_HIDDEN_LAYER_NEURONS = 64 #for the hidden layers
    NUM_OUTPUT_LAYER_NEURONS = 10 #placeholder
    
    MUTATION_RATE = 0.1 #probability that a parameter will be mutated in
    MUTATION_PERCENT = 0.1 #10% chance to mutate a gene in each generation
    
    def __init__(self, board_state):
        super().__init__() #creates an instance
        self.layers = nn.ModuleList() #makes an empty list containing the input, hidden, and output layers
        self.layers.append(nn.Linear( board_state + 1, AI_Agent.NUM_HIDDEN_LAYER_NEURONS, bias=False)) #creates input layer
        
        #creates the hidden layers
        for i in range(AI_Agent.NUM_HIDDEN_LAYERS - 1): 
            self.layers.append(nn.Linear(AI_Agent.NUM_HIDDEN_LAYER_NEURONS, AI_Agent.NUM_HIDDEN_LAYER_NEURONS, bias=False))
            
        # Output layer corresponds to speed and angular velocity
        self.layers.append(nn.Linear(AI_Agent.NUM_HIDDEN_LAYER_NEURONS, AI_Agent.NUM_OUTPUT_LAYER_NEURONS, False)) 
        #Number of output nodes should be x, 200 is placeholder

        for layer in self.layers:
            nn.init.constant_(layer.weight, rnd.randrange(0,1)) #gives each weight a starting value between 0 and 1

        self.eval() #stops the training


    #the output layer generation
    def forward(self, input): #original def forward(self, x, viewDistance)
        """ Feed input into the neural network and obtain movement information as output """
        x = torch.FloatTensor(input) #since all values between 0 and 1, no need for normalization
        
        with torch.no_grad():
            for i in range(AI_Agent.NUM_HIDDEN_LAYERS):
                x = F.relu(self.layers[i](x))
            x = self.layers[-1](x)
        
        return special.expit(x).tolist()
    
    #generate values random values to give to each bridge
    def mutation(self):
        #iterate through each layer in the network
        for layer in self.layers:
            #get all the neurons in the current layer
            weights = layer.weight.data
            
            #make selection based on mutation rate
            mutate_selections = torch.rand_like(weights) < AI_Agent.MUTATION_RATE
            
            #mutate the weight by changing the it by the muatation percent
            mutate_gen = torch.rand_like(weights) * AI_Agent.MUTATION_PERCENT
            
            #multiply the original value by the mutation gen
            weights[mutate_selections] += mutate_gen[mutate_selections]