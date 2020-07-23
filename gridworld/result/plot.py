import matplotlib.pyplot as plt
import numpy as np
import os

figure = plt.figure()
ax1 = figure.add_subplot(121)
ax2 = figure.add_subplot(122)

DQN_q_mean = []
DQN_q_std = []
DQN_reward_mean = []
DQN_reward_std = []

k5_q_mean = []
k5_q_std = []
k5_reward_mean = []
k5_reward_std = []

k10_q_mean = []
k10_q_std = []
k10_reward_mean = []
k10_reward_std = []

DDQN_q_mean = []
DDQN_q_std = []
DDQN_reward_mean = []
DDQN_reward_std = []

avg_DDQN_q_mean = []
avg_DDQN_q_std = []
avg_DDQN_reward_mean = []
avg_DDQN_reward_std = []


# 20 epochs
n_epoch = 20
for i in range(1, n_epoch+1):
    # vanilla DQN
    DQN_q_tmp = []
    DQN_reward_tmp = []
    for directory in os.listdir('path to vanilla DQN result folder'):
        q_file_name = os.path.join(os.path.join('path to vanilla DQN result folder', directory), 'test_q_{}.npy'.format(i))
        reward_file_name = os.path.join(os.path.join('path to vanilla DQN result folder', directory), 'test_reward_{}.npy'.format(i))
        DQN_q = np.load(q_file_name)
        DQN_reward = np.load(reward_file_name)

        DQN_q_tmp.append(DQN_q)
        DQN_reward_tmp.append(DQN_reward)

    DQN_q_mean.append(np.mean(DQN_q_tmp))
    DQN_q_std.append(np.std(DQN_q_tmp))

    DQN_reward_mean.append(np.mean(DQN_reward_tmp))
    DQN_reward_std.append(np.std(DQN_reward_tmp))

    # k = 5
    k5_q_tmp = []
    k5_reward_tmp = []
    for directory in os.listdir('path to k = 5 averaged-DQN result folder'):
        q_file_name = os.path.join(os.path.join('path to k = 5 averaged-DQN result folder', directory), 'test_q_{}.npy'.format(i))
        reward_file_name = os.path.join(os.path.join('path to k = 5 averaged-DQN result folder', directory), 'test_reward_{}.npy'.format(i))
        k5_q = np.load(q_file_name)
        k5_reward = np.load(reward_file_name)

        k5_q_tmp.append(k5_q)
        k5_reward_tmp.append(k5_reward)

    k5_q_mean.append(np.mean(k5_q_tmp))
    k5_q_std.append(np.std(k5_q_tmp))

    k5_reward_mean.append(np.mean(k5_reward_tmp))
    k5_reward_std.append(np.std(k5_reward_tmp))

    # k = 10
    k10_q_tmp = []
    k10_reward_tmp = []
    for directory in os.listdir('path to k = 10 averaged-DQN result folder'):
        q_file_name = os.path.join(os.path.join('path to k = 10 averaged-DQN result folder', directory), 'test_q_{}.npy'.format(i))
        reward_file_name = os.path.join(os.path.join('path to k = 10 averaged-DQN result folder', directory), 'test_reward_{}.npy'.format(i))
        k10_q = np.load(q_file_name)
        k10_reward = np.load(reward_file_name)

        k10_q_tmp.append(k10_q)
        k10_reward_tmp.append(k10_reward)

    k10_q_mean.append(np.mean(k10_q_tmp))
    k10_q_std.append(np.std(k10_q_tmp))

    k10_reward_mean.append(np.mean(k10_reward_tmp))
    k10_reward_std.append(np.std(k10_reward_tmp))

    # DDQN
    DDQN_q_tmp = []
    DDQN_reward_tmp = []
    for directory in os.listdir('path to DDQN result folder'):
        q_file_name = os.path.join(os.path.join('path to DDQN result folder', directory), 'test_q_{}.npy'.format(i))
        reward_file_name = os.path.join(os.path.join('path to DDQN result folder', directory), 'test_reward_{}.npy'.format(i))
        DDQN_q = np.load(q_file_name)
        DDQN_reward = np.load(reward_file_name)

        DDQN_q_tmp.append(DDQN_q)
        DDQN_reward_tmp.append(DDQN_reward)

    DDQN_q_mean.append(np.mean(DDQN_q_tmp))
    DDQN_q_std.append(np.std(DDQN_q_tmp))

    DDQN_reward_mean.append(np.mean(DDQN_reward_tmp))
    DDQN_reward_std.append(np.std(DDQN_reward_tmp))

    # k = 10 DDQN
    avg_DDQN_q_tmp = []
    avg_DDQN_reward_tmp = []
    for directory in os.listdir('path to avg-DDQN result folder'):
        q_file_name = os.path.join(os.path.join('path to avg-DDQN result folder', directory), 'test_q_{}.npy'.format(i))
        reward_file_name = os.path.join(os.path.join('path to avg-DDQN result folder', directory), 'test_reward_{}.npy'.format(i))
        avg_DDQN_q = np.load(q_file_name)
        avg_DDQN_reward = np.load(reward_file_name)

        avg_DDQN_q_tmp.append(avg_DDQN_q)
        avg_DDQN_reward_tmp.append(avg_DDQN_reward)

    avg_DDQN_q_mean.append(np.mean(avg_DDQN_q_tmp))
    avg_DDQN_q_std.append(np.std(avg_DDQN_q_tmp))

    avg_DDQN_reward_mean.append(np.mean(avg_DDQN_reward_tmp))
    avg_DDQN_reward_std.append(np.std(avg_DDQN_reward_tmp))

# plot Q value
ax1.plot(np.arange(n_epoch)+1, DQN_q_mean, color='blue', label='vanilla DQN')
ax1.fill_between(np.arange(n_epoch)+1, np.array(DQN_q_mean)-np.array(DQN_q_std), np.array(DQN_q_mean)+np.array(DQN_q_std), color='lightblue')

ax1.plot(np.arange(n_epoch)+1, k10_q_mean, color='red', label='k = 10')
ax1.fill_between(np.arange(n_epoch)+1, np.array(k10_q_mean)-np.array(k10_q_std), np.array(k10_q_mean)+np.array(k10_q_std), color='salmon')

ax1.plot(np.arange(n_epoch)+1, k5_q_mean, color='green', label='k = 5')
ax1.fill_between(np.arange(n_epoch)+1, np.array(k5_q_mean)-np.array(k5_q_std), np.array(k5_q_mean)+np.array(k5_q_std), color='lightgreen')

ax1.plot(np.arange(n_epoch)+1, DDQN_q_mean, color='yellow', label='DDQN')
ax1.fill_between(np.arange(n_epoch)+1, np.array(DDQN_q_mean)-np.array(DDQN_q_std), np.array(DDQN_q_mean)+np.array(DDQN_q_std), color='tan')

ax1.plot(np.arange(n_epoch)+1, avg_DDQN_q_mean, color='purple', label='avg-DDQN k = 10')
ax1.fill_between(np.arange(n_epoch)+1, np.array(avg_DDQN_q_mean)-np.array(avg_DDQN_q_std), np.array(avg_DDQN_q_mean)+np.array(avg_DDQN_q_std), color='orchid')

# the optimal Q value
# 38 is computed by 2*(size-1)
optimal = [0.9**38 for _ in range(n_epoch)]
ax1.plot(np.arange(n_epoch)+1, optimal, linestyle=':', color='black', label='optimal Q')

# plot reward score
ax2.plot(np.arange(n_epoch)+1, DQN_reward_mean, color='blue', label='vanilla DQN')
ax2.fill_between(np.arange(n_epoch)+1, np.array(DQN_reward_mean)-np.array(DQN_reward_std), np.array(DQN_reward_mean)+np.array(DQN_reward_std), color='lightblue')

ax2.plot(np.arange(n_epoch)+1, k10_reward_mean, color='red', label='k = 10')
ax2.fill_between(np.arange(n_epoch)+1, np.array(k10_reward_mean)-np.array(k10_reward_std), np.array(k10_reward_mean)+np.array(k10_reward_std), color='salmon')

ax2.plot(np.arange(n_epoch)+1, k5_reward_mean, color='green', label='k = 5')
ax2.fill_between(np.arange(n_epoch)+1, np.array(k5_reward_mean)-np.array(k5_reward_std), np.array(k5_reward_mean)+np.array(k5_reward_std), color='lightgreen')

ax2.plot(np.arange(n_epoch)+1, DDQN_reward_mean, color='yellow', label='DDQN')
ax2.fill_between(np.arange(n_epoch)+1, np.array(DDQN_reward_mean)-np.array(DDQN_reward_std), np.array(DDQN_reward_mean)+np.array(DDQN_reward_std), color='tan')

ax2.plot(np.arange(n_epoch)+1, avg_DDQN_reward_mean, color='purple', label='avg-DDQN k = 10')
ax2.fill_between(np.arange(n_epoch)+1, np.array(avg_DDQN_reward_mean)-np.array(avg_DDQN_reward_std), np.array(avg_DDQN_reward_mean)+np.array(avg_DDQN_reward_std), color='orchid')

ax1.set_title('Gridworld (20x20) Q value estimation')
ax2.set_title('Gridworld (20x20) score')

ax1.set_xlabel('epochs (100k steps)')
ax1.set_ylabel('Q value')

ax2.set_xlabel('epochs (100k steps)')
ax2.set_ylabel('score')

ax1.legend()
ax2.legend()
#plt.savefig('all_q.png')
plt.show()