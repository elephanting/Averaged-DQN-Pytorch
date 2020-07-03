import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F

class Env:
    def __init__(self, grid=(5, 5)):
        assert len(grid) == 2, "input x and y"
        assert grid[0] > 0 and grid[1] > 0, "input positive number"
        self.gridsize = grid
        self.goal = (grid[0]-1, grid[1]-1)
        self.position = np.zeros(2, dtype=int)

        # initialize grid
        self.grid = np.zeros((grid[0], grid[1]))
        self.grid[self.goal] = 2
        self.done = 0
    def reset(self, random_loc=True):
        '''
            observation: player: 1
                         reward: 2
                         else: 0
        '''
        
        self.grid = np.zeros(self.grid.shape)
        if random_loc is True:
            # random player position
            random_position = np.random.randint(self.gridsize[0]*self.gridsize[1]-1)
            self.position = np.array([random_position // self.gridsize[0], random_position % self.gridsize[1]], dtype=int)
        else:
            self.position = np.zeros(2, dtype=int)

        self.grid[self.position[0], self.position[1]] = 1
        self.grid[self.goal] = 2
        self.done = 0

        return self.grid
    
    def step(self, action, test=False):
        # action(up:0, down:1, right:2, left:3)
        original_position = self.position.copy()
        out_of_boundary = False
        if action == 0:
            if self.position[0] - 1 >= 0:
                self.position[0] = self.position[0] - 1
            else:
                pass
                #out_of_boundary = True        
        elif action == 1:
            if self.position[0] + 1 < self.gridsize[0]:
                self.position[0] = self.position[0] + 1
            else:
                pass
                #out_of_boundary = True
        elif action == 2:
            if self.position[1] + 1 < self.gridsize[1]:
                self.position[1] = self.position[1] + 1
            else:
                pass
                #out_of_boundary = True            
        elif action == 3:
            if self.position[1] - 1 >= 0:
                self.position[1] = self.position[1] - 1
            else:
                pass
                #out_of_boundary = True

        if self.position[0] == self.gridsize[0]-1 and self.position[1] == self.gridsize[1]-1:
            reward = 1
            done = 1
        elif out_of_boundary:
            #print(self.position[0])
            reward = -1
            done = 1
        else:
            reward = 0
            done = 0
            self.grid[self.position[0], self.position[1]] = 1
            self.grid[original_position[0], original_position[1]] = 0
        
        return self.grid, reward, done

if __name__ == '__main__':
    env = Env()
    state = env.reset(random_loc=False)
    state = env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(2)
    env.step(2)
    env.step(1)
    state = env.step(2)
    state = env.reset(random_loc=False)
    
    print(state)