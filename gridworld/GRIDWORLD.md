# Gridworld
The gridworld folder aims to experiment the overestimation phenomenon.
The optimal Q value of Gridworld is deterministic. We can see the overestimation phenomenon between DQN and Averaged-DQN.

## Usage
The usage of gridworld folder is similar to the main folder

#### Examples:
For vanilla DQN training:
```sh
python train.py
```
For averaged-DQN training:
```sh
python train.py --k 10
```
For averaged-DDQN training:
```sh
python train.py --k 10 --ddqn
```

## Arguments
| Argument      | Description   |
| ------------- | ------------- |
| --k           | average k Q values, k = 1 is equal to vanilla DQN |
| --ddqn        | perform Double-DQN |
| --size      | size of square gridworld |

## Optimal Q value
According to [DDQN paper](https://arxiv.org/pdf/1509.06461.pdf), the optimal Q value is computed by ***The ground truth averaged values are obtained by running
the best learned policies for several episodes and computing the actual cumulative rewards***. So optimal Q value will be

where gamma is discounted factor.

## Evaluation of DQN and Averaged-DQN
