import os
import sys
import gym
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet, QNet_more_layers
from memory import Memory_With_TDError
from tensorboardX import SummaryWriter
from config import env_name, gamma, initial_exploration, batch_size, update_target, log_interval, goal_score, device, replay_memory_capacity, lr, beta_start
from gym_brt.envs.qube_swingup_env import QubeSwingupEnv
from gym_brt.control import pd_control_policy

parser = argparse.ArgumentParser()
parser.add_argument('--save', action="store_true", default=False)
parser.add_argument('--sim', action="store_true", default=False)
parser.add_argument('--entropy', action="store_true", default=False)
parser.add_argument('--boltzmann', action="store_true", default=False)
parser.add_argument('--na', action="store", dest="na", type=int, default=6)
parser.add_argument('--temp', action="store", dest="temp", type=float, default=1)
parser.add_argument('--al', action="store", dest="al", type=float, default=3.0)
parser.add_argument('--dir', action="store", dest="dir", type=str, default="trash")


args = parser.parse_args()

NUMBER_OF_ACTIONS = args.na

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


def get_action(state, target_net):
    return target_net.get_action(state)


def main():

    if not (os.path.isdir("logs")):
        os.makedirs("logs")

    working_dir = "logs/" + args.dir
    if not (os.path.isdir(working_dir)):
        raise NameError(args.dir + " does not exist in dir logs")

    print(args)

    env = QubeSwingupEnv(use_simulator=args.sim, batch_size= 2048*4)

    num_inputs = env.observation_space.shape[0]
    num_actions = NUMBER_OF_ACTIONS
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = QNet(num_inputs, num_actions) if not args.new_net else QNet_more_layers(num_inputs, num_actions)
    net.load_state_dict(torch.load(working_dir + "/best_model.pth", map_location=torch.device(device)))
    net.to(device)
    net.eval()
    running_score = 0
    epsilon = 1.0
    steps = 0
    beta = beta_start
    loss = 0

    best_running_score = -1000

    for e in range(1):
        done = False

        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)
        
        while not done:
            steps += 1
            action = get_continuous_action(get_action(state, net))
            if np.abs(state[0][1].item()) < deg2rad(25):
                action = pd_control_policy(state.cpu().numpy()[0])[0]
            next_state, reward, done, info = env.step(action)
            reward = give_me_reward(info["alpha"], info["theta"])
            if args.sim: env.render()
            reward = give_me_reward(info["alpha"], info["theta"])
            if done:
                print(info)
                print("theta:" , info["theta"] * 180/np.pi)
            next_state = torch.Tensor(next_state).to(device)
            next_state = next_state.unsqueeze(0)

            score += reward
            state = next_state

        running_score = 0.99 * running_score + 0.01 * score
        print('{} episode | running_score: {:.2f} | score: {:.2f} | steps: {} '.format(e, running_score, score, steps))
    env.close()


if __name__=="__main__":
    main()
