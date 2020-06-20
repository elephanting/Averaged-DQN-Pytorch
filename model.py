import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import random

from ER import ReplayMemory

class Net(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

class DQN:
    def __init__(self, args, env):
        self._behavior_net = Net(env.observation_space.shape, env.action_space.n).to(args.device)
        if args.k > 1:
            # averaged-DQN
            self._target_net = [Net(env.observation_space.shape, env.action_space.n).to(args.device) for _ in range(args.k)]
        else:
            # vanilla DQN
            self._target_net = Net(env.observation_space.shape, env.action_space.n).to(args.device)
        
        # initialize target network
        if args.k > 1:
            for i in range(args.k):
                self._target_net[i].load_state_dict(self._behavior_net.state_dict())
        else:
            self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._optimizer = torch.optim.RMSprop(self._behavior_net.parameters(), lr=args.lr, momentum=args.momentum)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.action_space_n = env.action_space.n
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq
        self.idx = 0 # determine which target network should be updated
        self.k = args.k
        self.ddqn = args.ddqn
        self.criterion = torch.nn.MSELoss()

        # some misc information
        self.epoch = 0
        self.total_steps = 0

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        if epsilon > random.random():
            action = action_space.sample()
        else:
            with torch.no_grad():
                q_value = self._behavior_net(torch.Tensor(state).unsqueeze(0).to(self.device))
                action = int(torch.argmax(q_value))
        return action

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, reward, next_state, done)

    def update(self, total_steps):
        if total_steps % self.freq == 0:            
            self._update_behavior_network(self.gamma)            
        if total_steps % self.target_freq == 0:
            self._update_target_network()
            self.idx += 1

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)
        
        q_value = self._behavior_net(state)
        q_value = torch.gather(q_value, dim=1, index=action.long())
        with torch.no_grad():
            if self.ddqn:
                q_next = self._behavior_net(next_state)
                action_next = torch.argmax(q_next, dim=1).unsqueeze(-1)
                if self.k > 1:
                    qs = torch.zeros(self.batch_size, self.action_space_n).to(self.device)
                    for i in range(self.k):
                        qs += self._target_net[i](next_state)
                    qs /= self.k
                else:
                    qs = self._target_net(next_state)
                q_next = torch.gather(qs, dim=1, index=action_next)
                q_target = reward + gamma * q_next * (1 - done)
            else:
                if self.k > 1:
                    q_next = torch.zeros(self.batch_size, self.action_space_n).to(self.device)
                    for i in range(self.k):
                        q_next += self._target_net[i](next_state)
                    q_next /= self.k
                else:
                    q_next = self._target_net(next_state)
                q_next = torch.max(q_next, dim=1).values.unsqueeze(-1)
                q_target = reward + gamma * q_next * (1 - done)
        loss = self.criterion(q_value, q_target)

        # optimize        
        self._optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()
        
    def _update_target_network(self):
        '''update target network by copying from behavior network'''        
        if self.k > 1:
            self._target_net[self.idx % self.k].load_state_dict(self._behavior_net.state_dict())
        else:
            self._target_net.load_state_dict(self._behavior_net.state_dict())        

    def save(self, model_path):
        if self.k > 1:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': [self._target_net[i].state_dict() for i in range(self.k)],
                    'optimizer': self._optimizer.state_dict(),
                    'epoch': self.epoch,
                    'step': self.total_steps,
                    'idx': self.idx
                }, model_path)
        else:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                    'epoch': self.epoch,
                    'step': self.total_steps
            }, model_path)
        self._memory.save(self.epoch)

    def load(self, model_path, test=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        self._optimizer.load_state_dict(model['optimizer'])
        self.epoch = model['epoch']
        self.total_steps = model['step']
        if self.k > 1:
            for i in range(self.k):
                self._target_net[i].load_state_dict(model['target_net'][i])
            self.idx = model['idx']
        else:
            self._target_net.load_state_dict(model['target_net'])
        
        if test is False:
            self._memory.load(self.epoch)