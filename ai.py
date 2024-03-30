import torch
import torch.nn as nn
import torch.nn.functional as F

import random as rnd
from scipy import special

class AI_Agent(nn.Module):
    
    NUM_HIDDEN_LAYERS = 2 # Minimum of 1 hidden layer, otherwise change init
    NUM_HIDDEN_LAYER_NEURONS = 64 #for the hidden layers
    NUM_OUTPUT_LAYER_NEURONS = 10 #default
    
    MUTATION_RATE = 0.1 #probability that a parameter will be mutated in
    MUTATION_PERCENT = 0.1 #10% chance to mutate a gene in each generation
    
    def __init__(self, outputLength, dataTxt):
        super().__init__() #creates an instance
        self.layers = nn.ModuleList() #makes an empty list containing the input, hidden, and output layers
        #Input will be all vertices, all bridges, and current resources
        #54 vertices + 72 bridges + 5 resources + 5 development cards
        inputLength = 54 + 72 + 5 + 5
        self.outputLength = outputLength
        self.layers.append(nn.Linear( inputLength, AI_Agent.NUM_HIDDEN_LAYER_NEURONS, bias=False)) #creates input layer
        
        #creates the hidden layers
        for i in range(AI_Agent.NUM_HIDDEN_LAYERS - 1): 
            self.layers.append(nn.Linear(AI_Agent.NUM_HIDDEN_LAYER_NEURONS, AI_Agent.NUM_HIDDEN_LAYER_NEURONS, bias=False))
            
        # Output layer corresponds to speed and angular velocity
        self.layers.append(nn.Linear(AI_Agent.NUM_HIDDEN_LAYER_NEURONS, outputLength, bias=False)) 
        #Number of output nodes should be x, 200 is placeholder

        if dataTxt == None:
            for layer in self.layers:
                nn.init.uniform_(layer.weight, -0.4, 0.4)  # Initialize weights between -1 and 1
                #Gives each weight a starting value between 0.2 and 0.8
                #This helps prevent exploding or vanishing gradients
        else:
            #Get dataTxt to write the connection data
            pass

        self.eval() #stops the training


    #the output layer generation
    def forward(self, input_data): #original def forward(self, x, viewDistance)
        """ Feed input into the neural network and obtain movement information as output """
        x = torch.FloatTensor(input_data) #since all values between 0 and 1, no need for normalization
        
        with torch.no_grad():
            for i in range(AI_Agent.NUM_HIDDEN_LAYERS):
                x = F.relu(self.layers[i](x))
            x = self.layers[-1](x)
        
        return special.expit(x).tolist()
    
    #generate values random values to give to each bridge
    def mutation(self):
        #iterate through each layer in the network
        for layer in self.layers:
            
            #make selection based on mutation rate
            mutate_selections = torch.rand_like(layer.weight.data) < AI_Agent.MUTATION_RATE
            
            #mutate the weight by changing the it by the muatation percent
            mutate_gen = torch.rand_like(layer.weight.data) * AI_Agent.MUTATION_PERCENT
            
            #multiply the original value by the mutation gen
            layer.weight.data[mutate_selections] += mutate_gen[mutate_selections]
