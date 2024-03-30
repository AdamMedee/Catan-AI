import torch
import torch.nn as nn
import torch.nn.functional as F
import os
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
       
        
        for layer in self.layers:
            # Roughly using xavier initialization
            # This helps prevent exploding or vanishing gradients
            nn.init.uniform_(layer.weight, -0.3, 0.3)

        if dataTxt != None:
            self.readWeights(dataTxt)
               

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
    def mutate(self):
        #iterate through each layer in the network
        for layer in self.layers:
            
            #make selection based on mutation rate
            mutate_selections = torch.rand_like(layer.weight.data) < AI_Agent.MUTATION_RATE
            
            #mutate the weight by changing the it by the muatation percent
            mutate_gen = torch.rand_like(layer.weight.data) * AI_Agent.MUTATION_PERCENT
            
            #multiply the original value by the mutation gen
            layer.weight.data[mutate_selections] += mutate_gen[mutate_selections]

    # This function will take a file of weights, and write this neural networks weights to the file
    def writeWeights(self, filename):
        """takes itself, and a txt file to output to"""
        file = open(filename, "w")
        
        for index, layer in enumerate(self.layers):
            weights = layer.weight.data.tolist()
            file.write(f"Layer {index}\n")
            file.write("Weights:")
            file.write(', '.join(str(w) for w in weights)+'\n')
            file.write('\n')

        file.close()

    # This function will take a file of weights, and write that file to the neural network
    def readWeights(self, filename):
        """load weights from a text file into agent's neural net"""
        if os.path.exists(filename):
            lines = open(filename, 'r').readlines()
            index = -1
            for line in  lines:
                if line.startswith("Layer"):
                    index += 1
                elif line.startswith("Weights"):
                    weights_txt = line.split(":")[-1].strip().replace("[", "").replace("]", "")
                    weights = list(map(float, weights_txt.split(',')))

                    # Determine the dimensions of the weight matrix
                    num_neurons_in_current_layer = self.layers[index].weight.data.size(0)
                    num_neurons_in_next_layer = self.layers[index].weight.data.size(1)

                    # Reshape the flat list of weights into a two-dimensional tensor
                    weights_tensor = torch.FloatTensor(weights).view(num_neurons_in_current_layer, num_neurons_in_next_layer)

                    # Assign the reshaped weights to the weight.data attribute of the current layer
                    self.layers[index].weight.data = weights_tensor
        else:
            pass

    
