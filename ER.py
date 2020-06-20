from collections import deque
import random
import torch
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
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))

    def save(self, epoch):
        print('Saving ER buffer...')
        with open('model/buffer_{}'.format(epoch), 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, epoch):
        print('Loading ER buffer...')
        with open('model/buffer_{}'.format(epoch), 'rb') as f:
            self.buffer = pickle.load(f)