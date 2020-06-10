import numpy as np
import matplotlib.pyplot as plt

reward = np.load('reward.npy')
idx = np.load('idx.npy')


plt.plot(np.arange(len(reward)), reward)
plt.show()