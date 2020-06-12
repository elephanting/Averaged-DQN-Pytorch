"""
    This file is copied/apdated from https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb
"""

import math, random

import gym
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt
from collections import deque
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# replay buffer
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
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

def compute_td_loss(batch_size, target, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action     = torch.LongTensor(action).to(device)
    reward     = torch.FloatTensor(reward).to(device)
    done       = torch.FloatTensor(done).to(device)

    q_values   = model(state)
    
    if args.average:
        # Averaged-DQN
        total_q = torch.zeros(batch_size, env.action_space.n).to(device)
        for i in range(args.k):
            total_q += target[i](next_state)
        next_q_values = total_q / args.k
    else:
        # vanilla DQN
        next_q_values = target(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - expected_q_value.data).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--average', action='store_true', help='perform Averaged-DQN')
    parser.add_argument('--k', type=int, default=10, help='if perform Averaged-DQN, average k target action values')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--path', type=str, help='resume training model path')
    parser.add_argument('--checkpoint', action='store_true', help='save DQN model every 1 million frames')

    # hyperparameters, default settings are referd to the Averaged-DQN paper
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor')
    parser.add_argument('--ER', type=int, default=10000000, help='Experience Replay buffer size')
    parser.add_argument('--update', type=int, default=10000, help='update target network every x frames')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--epsilon', type=int, default=10000000, help='epsilon greedy algo, decreasing linearly from 1 to 0.1 over x steps')
    parser.add_argument('--total', type=int, default=1000000000, help='total training frames')
    args = parser.parse_args()

    num_frames = args.total
    batch_size = args.batch
    gamma      = args.discount

    '''
    a = torch.Tensor([5])
    b = torch.Tensor([4])
    d = torch.Tensor([6])
    #b.requires_grad = True
    a.requires_grad = True
    d.requires_grad = True
    e = a * d
    #a.detach()
    c = a * b
    print(c.requires_grad)
    c.backward()
    print(b.grad.data)
    '''


    # Atari Environment
    env_id = "BreakoutNoFrameskip-v4"
    env    = make_atari(env_id)
    env    = wrap_deepmind(env)
    env    = wrap_pytorch(env)

    model = DQN(env.observation_space.shape, env.action_space.n).to(device).train()
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.average:
        Qs = []
        for _ in range(args.k):
            copy_model = DQN(env.observation_space.shape, env.action_space.n).to(device).eval()
            copy_model.load_state_dict(model.state_dict())
            Qs.append(copy_model)
    else:
        target_model = DQN(env.observation_space.shape, env.action_space.n).to(device).eval()
        target_model.load_state_dict(model.state_dict())

    if args.resume:
        checkpoint = torch.load(args.path)
        if args.average:
            for i in range(args.k):
                Qs[i].load_state_dict(checkpoint['Qs'][i])
        else:
            # vanilla DQN
            target_model.load_state_dict(checkpoint['target'])
        
        episode_reward = 0
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_frame = checkpoint['frame_idx']
        frame_done = checkpoint['frame_done']
        q_idx = checkpoint['q_idx']
        all_rewards = checkpoint['all_rewards']
    else:
        # train from scratch
        all_rewards = []
        frame_done = []
        q_idx = 0
        episode_reward = 0
        start_frame = 1

    replay_initial = 10000
    replay_buffer = ReplayBuffer(args.ER)
    
    # Epsilon greedy exploration
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = args.epsilon

    epsilon_by_frame = lambda frame_idx, replay_start_time: 1 - 0.9*min(replay_start_time, args.epsilon)/args.epsilon

    # start training 
    state = env.reset()
    for frame_idx in range(start_frame, num_frames + 1):
        #env.render()
        epsilon = epsilon_by_frame(frame_idx, max([frame_idx-replay_initial, 0]))
        action = model.act(state, epsilon)
        
        next_state, reward, done, _ = env.step(action)
        reward = np.clip(reward, -1, 1)
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            frame_done.append(frame_idx)
            episode_reward = 0
            
        if len(replay_buffer) > replay_initial:
            if args.average:
                loss = compute_td_loss(batch_size, Qs, replay_buffer)
            else:
                loss = compute_td_loss(batch_size, target_model, replay_buffer)

        if frame_idx % 10000 == 0:
            if args.average:
                idx = q_idx % (args.k)
                q_idx += 1
                Qs[idx].load_state_dict(model.state_dict())
            else:
                target_model.load_state_dict(model.state_dict())
            
            if frame_idx % 100000 == 0:
                print(frame_idx)
                print(np.mean(all_rewards[-100:]))

                if args.checkpoint and frame_idx % 500000 == 0:
                    # save model
                    if args.average:
                        # Averaged-DQN model
                        torch.save({
                                    'Qs': [Qs[i].state_dict() for i in range(args.k)],
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'all_rewards': all_rewards,
                                    'frame_idx': frame_idx,
                                    'frame_done': frame_done,
                                    'q_idx': q_idx                
                                    }, './model/frame_{}.tar'.format(frame_idx))
                    else:
                        # vanilla DQN model
                        torch.save({
                                    'target': target_model.state_dict(),
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'all_rewards': all_rewards,
                                    'frame_idx': frame_idx,
                                    'frame_done': frame_done,
                                    'q_idx': q_idx                    
                                    }, './model/frame_{}.tar'.format(frame_idx))
