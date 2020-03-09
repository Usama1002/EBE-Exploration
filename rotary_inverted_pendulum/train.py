import os
import sys
import gym
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
from memory import Memory_With_TDError
from tensorboardX import SummaryWriter
from config import env_name, gamma, initial_exploration, batch_size, update_target, log_interval, goal_score, device, replay_memory_capacity, lr, beta_start
from gym_brt.envs.qube_swingup_env import QubeSwingupEnv
import time
import scipy.stats as sc

parser = argparse.ArgumentParser()
parser.add_argument('--save', action="store_true", default=False)
parser.add_argument('--entropy', action="store_true", default=False)
parser.add_argument('--boltzmann', action="store_true", default=False)
parser.add_argument('--na', action="store", dest="na", type=int, default=6)
parser.add_argument('--temp', action="store", dest="temp", type=float, default=1)
parser.add_argument('--al', action="store", dest="al", type=float, default=3.0)
parser.add_argument('--dir', action="store", dest="dir", type=str, default="trash")
parser.add_argument('--rp', action="store", dest="rp", type=float, default=1)
parser.add_argument('--e', action="store", dest="e", type=int, default=1000)

args = parser.parse_args()


NUMBER_OF_ACTIONS = args.na

def softmax(x):
    """Compute softmax values for each sets of scores in x using 'max' trick."""
    max_x = max(x)
    x = [i - max_x for i in x]
    to_return = np.exp(x) / np.sum(np.exp(x), axis=0)
    return to_return

def calculate_entropy(action_values):
    action_values = action_values.data.cpu().numpy()[0]
    base = np.shape(action_values)[0]
    action_probabilities = softmax(action_values)
    entropy = sc.entropy(action_probabilities, base = base)
    return entropy

def deg2rad(angle):
    return (angle * np.pi / 180.0)

def give_me_reward(alpha, theta):
    if np.abs(theta) > deg2rad(90):
        return -100
    reward = 1 - ((0.8 * np.abs(alpha) + 0.2 * np.abs(theta)) / np.pi)
    if np.abs(alpha) < deg2rad(25):
        reward = 100 + reward
    return reward

def get_continuous_action(discrete_action, minusL = -args.al, plusL = args.al, N = NUMBER_OF_ACTIONS):
    continuous_action = minusL + discrete_action * (plusL - minusL)/(N-1)
    return np.clip(continuous_action, minusL, plusL)


def get_action(state, target_net, epsilon, use_entropy, use_boltzmann):
    if args.boltzmann:
        action_values = target_net.get_qValues(state)
        return np.random.choice(action_values.shape[1], p = softmax(action_values.data.cpu().numpy()[0]/args.temp))      # boltzmann exploration
    
    if args.entropy:
        action_values = target_net.get_qValues(state)
        epsilon = calculate_entropy(action_values)

    if np.random.rand() <= epsilon:
        return np.random.randint(0, NUMBER_OF_ACTIONS)
    else:
        return target_net.get_action(state)


def update_target_model(online_net, target_net):
    # Target -> Net
    target_net.load_state_dict(online_net.state_dict())


def main():

    if not (os.path.isdir("logs")):
        os.makedirs("logs")


    if (args.entropy and args.boltzmann):
        raise ValueError("Entropy as well as Boltzmann set.")

    print(args)

    working_dir = "logs/" + args.dir
    if not (os.path.isdir(working_dir)):
        os.mkdir(working_dir)

    env = QubeSwingupEnv(use_simulator=True)

    num_inputs = env.observation_space.shape[0]
    num_actions = NUMBER_OF_ACTIONS
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)

    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    writer = SummaryWriter(working_dir)

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory_With_TDError(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    beta = beta_start
    loss = 0
    training_started = False

    best_running_score = -1000

    for e in range(args.e):
        done = False

        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)
        start_time = time.time()
        
        while not done:
            steps += 1
            action = get_action(state, target_net, epsilon, use_entropy=args.entropy, use_boltzmann=args.boltzmann)
            next_state, reward, done, info = env.step(get_continuous_action(action))

            reward = give_me_reward(info["alpha"], info["theta"])

            next_state = torch.Tensor(next_state).to(device)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            action_one_hot = np.zeros(NUMBER_OF_ACTIONS)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

            if steps > initial_exploration:
                if not training_started:
                    print("---------------- training started ---------------")
                    training_started = True
                epsilon -= 0.000005
                epsilon = max(epsilon, 0.1)
                beta += 0.000005
                beta = min(1, beta)

                batch, weights = memory.sample(batch_size, online_net, target_net, beta)
                loss = QNet.train_model(online_net, target_net, optimizer, batch, weights, device)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        end_time = time.time()
        running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f} | beta: {:.2f}'.format(
                e, running_score, epsilon, beta))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > best_running_score and args.save:
            torch.save(online_net.state_dict(), working_dir + "/best_model.pth")
            best_running_score = running_score


if __name__=="__main__":
    main()
