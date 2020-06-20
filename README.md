# Averaged-DQN
Paper: [Averaged-DQN: Variance Reduction and Stabilization
for Deep Reinforcement Learning](https://arxiv.org/pdf/1611.01929.pdf)

## Installation

ubuntu16.04:
```sh
conda env create -f environment.yml
```
## Usage
First activate virtual environment
```sh
conda activate gym
```
#### Examples:
For vanilla DQN training:
```sh
python DQN.py --checkpoint 
```
For averaged-DQN training:
```sh
python DQN.py --checkpoint --k 10
```
For averaged-DDQN training:
```sh
python DQN.py --checkpoint --k 10 --ddqn
```
## Arguments
| Argument      | Description   |
| ------------- | ------------- |
| --checkpoint  | save DQN model every epoch  |
| --k           | average k Q values, k = 1 is equal to vanilla DQN |
| --ddqn        | perform Double-DQN |
| --resume      | resume training |
| --model        | model path used in resume training |
