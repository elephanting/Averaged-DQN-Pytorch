"""
    This file is copied/apdated from https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb
"""

import math, random

import gym
import numpy as np
import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--average', action='store_true', help='perform Averaged-DQN')
parser.add_argument('--k', type=int, default=1, help='if perform Averaged-DQN, average k action values')
args = parser.parse_args()

# use CUDA
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# Atari Environment
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

env_id = "Breakout-v0"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)


# replay buffer
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)


# Deep Q Network
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
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
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

model = DQN(env.observation_space.shape, env.action_space.n)

if USE_CUDA:
    model = model.cuda()

if args.average:
    Qs = []
    for _ in range(args.k-1):
        copy_model = DQN(env.observation_space.shape, env.action_space.n).cuda()
        copy_model.load_state_dict(model.state_dict())
        Qs.append(copy_model)

target_model = DQN(env.observation_space.shape, env.action_space.n).cuda()
target_model.load_state_dict(model.state_dict())

def compute_td_loss(batch_size, idx):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    with torch.no_grad():
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    if args.average:
        # Averaged-DQN
        total_q = model(state)
        with torch.no_grad():
            for i in range(args.k-1):
                total_q += Qs[i](state)
        q_values = total_q / args.k
    else:
        # normal DQN
        q_values      = model(state)
        
    next_q_values = model(next_state)


    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if args.average:
        index = frame_idx % (args.k-1)
        Qs[index].load_state_dict(model.state_dict())

    return loss

def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    #plt.show()
       
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Epsilon greedy exploration
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# training
losses = []
all_rewards = []
frame_done = []
episode_reward = 0

num_frames = 1400000
batch_size = 32
gamma      = 0.99

state = env.reset()
episode = 0
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        #env.render()
        all_rewards.append(episode_reward)
        frame_done.append(frame_idx)
        episode_reward = 0
        episode += 1
        #if episode % 10 == 0:
        #    print(episode)
        
    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(batch_size, frame_idx)
        #losses.append(loss.item())
        
    if frame_idx % 100000 == 0:
        #plot(frame_idx, all_rewards, losses)
        np.save('idx.npy', frame_idx)
        np.save('reward.npy', all_rewards)
        print(frame_idx)
        print(all_rewards[-1])

    if frame_idx % 100 == 0:
        target_model = copy.deepcopy(model)