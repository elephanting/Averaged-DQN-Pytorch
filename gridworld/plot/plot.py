import matplotlib.pyplot as plt
import numpy as np
import os

dqn_res = []
dqn_std = []
k10_res = []
k10_std = []
ddqn_std = []
ddqn_res = []
k30_res = []
k30_std = []
for i in range(1, 6):
    dqn_mean = []
    ddqn_mean = []
    k10_mean = []
    k30_mean = []
    for j, dire in enumerate(os.listdir('./')):
        #print(dire)
        if os.path.isdir(dire) and (dire == '1' or dire == '2' or dire == '3'):
            lst = os.listdir(dire)
            for filename in lst:
                epoch = filename.split('_')[1].split('.')[0]
                if i == int(epoch):
                    data = np.load(os.path.join(dire, filename))
                    dqn_mean.append(data)
        elif os.path.isdir(dire) and (dire == '4' or dire == '5' or dire == '6'):
            lst = os.listdir(dire)
            for filename in lst:
                epoch = filename.split('_')[1].split('.')[0]
                if i == int(epoch):
                    data = np.load(os.path.join(dire, filename))
                    ddqn_mean.append(data)
        elif os.path.isdir(dire) and (dire == '7' or dire == '8' or dire == '9'):
            lst = os.listdir(dire)
            for filename in lst:
                epoch = filename.split('_')[1].split('.')[0]
                if i == int(epoch):
                    data = np.load(os.path.join(dire, filename))
                    k10_mean.append(data)
        elif os.path.isdir(dire) and (dire == '10' or dire == '11' or dire == '12'):
            # k30
            lst = os.listdir(dire)
            for filename in lst:
                epoch = filename.split('_')[1].split('.')[0]
                if i == int(epoch):
                    data = np.load(os.path.join(dire, filename))
                    k30_mean.append(data)



    dqn_res.append(np.mean(dqn_mean))
    dqn_std.append(np.std(dqn_mean))
    ddqn_res.append(np.mean(ddqn_mean))
    ddqn_std.append(np.std(ddqn_mean))
    k10_res.append(np.mean(k10_mean))
    k10_std.append(np.std(k10_mean))
    k30_res.append(np.mean(k30_mean))
    k30_std.append(np.std(k30_mean))
    
figure = plt.figure()
ax = figure.add_subplot(1, 1, 1)
ax.set_title('Gridworld 5x5')
ax.set_ylabel('value estimation')
ax.set_xlabel('epochs (10k frames)')

ax.plot(np.arange(5)+1, dqn_res, label='DQN', color='blue')
ax.fill_between(np.arange(5)+1, np.array(dqn_res)-np.array(dqn_std), np.array(dqn_res)+np.array(dqn_std), color='lightskyblue')

ax.plot(np.arange(5)+1, ddqn_res, label='avg-DDQN', color='red')
ax.fill_between(np.arange(5)+1, np.array(ddqn_res)-np.array(ddqn_std), np.array(ddqn_res)+np.array(ddqn_std), color='lightskyblue')

ax.plot(np.arange(5)+1, k10_res, label='k=10')
ax.fill_between(np.arange(5)+1, np.array(k10_res)-np.array(k10_std), np.array(k10_res)+np.array(k10_std))

ax.plot(np.arange(5)+1, k30_res, label='k=30')
ax.fill_between(np.arange(5)+1, np.array(k30_res)-np.array(k30_std), np.array(k30_res)+np.array(k30_std))

plt.legend()
plt.savefig('value_both.png')
plt.show()