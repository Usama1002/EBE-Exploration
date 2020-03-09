This is the code for our experiments on Qube Servo 2 platform in our paper Entropy-Based Exploration for Reinforcement Learning. Basic skeletion of code is taken from [here](https://github.com/g6ling/Reinforcement-Learning-Pytorch-Cartpole/tree/master/rainbow).


# Prerequisites
	- python3.6
	- Pytorch 1.4.0

# Simulator and Hardware Driers

We use OpenAI Gym wrapper for the Quanser Qube Servo 2 by [Blue River Tech](https://github.com/BlueRiverTech/quanser-openai-driver). Please follow [the instructions here](https://github.com/BlueRiverTech/quanser-openai-driver) to setup the required drivers and simulator.

# How to Run

## Training

### Entropy-Based Exploration (EBE)

Use the follwoing command to run experiment with EBE.

```
	python3.6 train.py --dir "save_dir"  --save --entropy
```

### Boltzmann Exploration

Use the follwoing command to run experiment with Boltzmann exploration.

```
	python3.6 train.py --dir "save_dir"  --save --boltzmann
```

### ε-greedy Exploration

Use the follwoing command to run experiment with ε-greedy exploration.

```
	python3.6 train.py --dir "save_dir"  --save
```

# Testing

For testing, the flag `--dir` represents the name of the directory that contains `best_model.pth` resulted from training process. Note that this directory is inside the `logs/` directory.
## Testing on Simulator

Use the following command to run test script on simulator.
```
python3.6 test.py --dir /dir_name/containing/best_model.pth/in/logs/folder --sim
```

## Testing on Hardware Platform

Use the following command to run test script on hardware Quaser Qube Servo 2.

```
python3.6 test.py --dir /dir_name/containing/best_model.pth/in/logs/folder
```

