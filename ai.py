from torch import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Agent:
    def __init__(self):
        #initialize the agent
        self.games = 0
    
    #get the current board state
    #takes in a list of 1080 elements, each element is a list of 5 values that are either 0 or 1
    #returns a list of 54 numbers, and each value in the list is either 0 or 1 indicating hexes that
    #settlements/cities can be built on
    def get_state(self, board_state):
        # Convert the input board state to a numpy array for easier manipulation
        board_state_np = np.array(board_state)
        
        #array of 54 numbers
        buildable = []
        #iterate over every element in the list given
        i = 0
        curr_element = 0
        while i < 54: #the total vetrices on the board
            j = 0
            value = 0 #1 or 0
            while j < 20: #
                 #do somethin with current element in the board state list
                 curr_element += 1
            buildable.append(value)
            
        return buildable #return the list of 54 numbers
    
    def get_action(self): 
        pass