from collections import deque
import random
import torch
import numpy as np
import pickle

class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        # (state, action, reward, next_state, done)
        self.buffer.append((state, [action], [reward], next_state, [done]))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        state = torch.FloatTensor(np.float32(state)).to(device)
        action = torch.LongTensor(action).to(device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)
        
        return state, action, reward, next_state, done

    def save(self, epoch):
        print('Saving ER buffer...')
        with open('model/buffer_{}'.format(epoch), 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, epoch):
        print('Loading ER buffer...')
        with open('model/buffer_{}'.format(epoch), 'rb') as f:
            self.buffer = pickle.load(f)