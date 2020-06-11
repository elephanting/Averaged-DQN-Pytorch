# Averaged-DQN
Paper: [Averaged-DQN: Variance Reduction and Stabilization
for Deep Reinforcement Learning](https://arxiv.org/pdf/1611.01929.pdf)

The code is copied/apdated from https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb

## Installation

ubuntu16.04:
```sh
conda env create -f environment.yml
```
## Usage
First activate virtual environment
```sh
conda activate gym_env
```
#### Examples:
For vanilla DQN training:
```sh
python DQN.py --checkpoint 
```
For averaged-DQN training:
```sh
python DQN.py --checkpoint --average --k 10
```
## Arguments
| Argument      | Description   |
| ------------- | ------------- |
| --checkpoint  | save DQN model every 1 million frames  |
| --average     | perform averaged-DQN training  |
| --k           | average k Q values |
| --resume      | resume training |
| --path        | model path used in resume training |
