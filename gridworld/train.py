import argparse
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from env import Env
from model import DQN

def init_buffer(env, agent):
    '''
        initialize all possible transitions into buffer
    '''
    n_states = env.gridsize

    # there are row*col-1 possibible agent location
    for row in range(n_states[0]):
        for col in range(n_states[1]):
            for action in range(4):
                if row == n_states[0]-1 and col == n_states[1]-1:
                    break

                # action(up:0, down:1, right:2, left:3)
                state = env.set_agent_loc(row, col)
                next_state, reward, done = env.step(action)
                agent.append(state, action, reward, next_state, done)

def train(args, env, agent, writer=None):
    action_space = 4
    total_steps = 0
    start_epoch = 1
    replay_initial = 0
    epsilon_by_steps = lambda steps, replay_start_time: 1 - (1-args.eps_min) * min(replay_start_time, args.epsilon)/args.epsilon

    # initialize ER buffer
    init_buffer(env, agent)

    if args.resume:
        if args.model is None:
            raise NameError('Please input model path')
        agent.load(args.model)
        total_steps = agent.total_steps
        start_epoch = agent.epoch + 1

    for epoch in range(start_epoch, args.epochs+1):
        print('Start Training')
        total_reward = 0
        state = env.reset(random_loc=False)
        rewards = []

        for t in tqdm(range(1, args.steps+1)):
            # select action
            epsilon = epsilon_by_steps(total_steps, max([total_steps-replay_initial, 0]))
            action = agent.select_action(state, epsilon, action_space)
            # execute action
            next_state, reward, done = env.step(action)
            
            agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                #writer.add_scalar('Train/Episode Reward', total_reward, total_steps)
                state = env.reset(random_loc=False)
                #print(total_reward)
                rewards.append(total_reward)
                total_reward = 0

            if t % 1000 == 0:
                if len(rewards) == 0:
                    rewards = total_reward
                tqdm.write('epoch: {}, total steps: {}, average reward: {:.2f}, epsilon: {:.2f}'.format(epoch, total_steps, np.mean(rewards), epsilon))
                rewards = []
        
        if args.checkpoint:
            agent.epoch = epoch
            agent.total_steps = total_steps
            agent.save('model/weight.tar')
        
        # every test need about 30 mins
        test(args, env, agent, epoch, writer)

def test(args, env, agent, epoch, writer):
    action_space = 4
    rewards = []
    total_reward = 0
    state = env.reset(random_loc=False)

    mean_q_val = agent.get_mean_q_val()

    for step in tqdm(range(args.test_steps)):
        action = agent.select_action(state, 0, action_space)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
    
        if done:
            #writer.add_scalar('Test/Episode Reward', total_reward, step)
            rewards.append(total_reward)
            total_reward = 0
            state = env.reset(random_loc=False)
    
    # last element of rewards is mean Q value
    print(rewards)

    print('Average Testing Reward:', np.mean(rewards))
    print('Average Q value: {:.2f}'.format(mean_q_val))
    np.save('result/test_{}.npy'.format(epoch), mean_q_val.cpu().numpy())

if __name__ == '__main__':
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--checkpoint', action='store_true', help='save model every epoch, ER buffer will only be saved as one file')
    parser.add_argument('--logdir', default='log/dqn')
    parser.add_argument('--size', type=int, default=20)
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--steps', default=100000, type=int, help='steps per epoch')
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--momentum', default=0.95, type=float)
    parser.add_argument('--epsilon', default=10000, type=int, help='decay steps from eps_max to eps_min')
    parser.add_argument('--eps_min', default=.05, type=float)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_steps', type=int, default=1140)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_epsilon', default=0, type=float)
    # average
    parser.add_argument('-k', '--k', type=int, default=1, help='number of average target network, if k = 1, perform vanilla DQN')
    # resume training
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('-m', '--model', default=None, help='model path')
    # DDQN
    parser.add_argument('--ddqn', action='store_true', help='perform Double-DQN')

    args = parser.parse_args()

    env = Env((args.size, args.size))
    args.capacity = (args.size*args.size-1)*4

    agent = DQN(args, env)
    train(args, env, agent)