
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
import os
import math
import scipy.stats as sc
import tensorflow as tf
import argparse
import warnings

# warnings.filterwarnings('error')

BEST_SCORE = -100000
my_global_step = 1
parser = argparse.ArgumentParser()
parser.add_argument('--entropy', action='store_true', help='whether to use entropy for exploration')
parser.add_argument('--bz', action='store_true', help='whether to use entropy for exploration')
parser.add_argument('--train', action='store_true', help='train the agent')
parser.add_argument('--test', action='store_true', help='test the agent')
parser.add_argument('--save_dir', type=str, default='./trash/')
parser.add_argument('--env', type=str, default='simpler_basic')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_steps_per_epoch', type=int, default=5000)
parser.add_argument('--replay_memory_size', type=int, default=50000)
parser.add_argument('--power', type=int, default=1)
parser.add_argument('--clip', action='store_true', help='clip')
parser.add_argument('--clip_value', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_eps_upto', type=float, default=0.1)
parser.add_argument('--decay_eps_upto', type=float, default=0.6)
parser.add_argument('--exponential', action='store_true', help='exponential decay')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag = tag, simple_value = value)
        self.writer.add_summary(summary, global_step = global_step)
        self.writer.flush()

tensorboard = Tensorboard(logdir=args.save_dir + '/tensorboard/')
# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
epochs = args.epochs
learning_steps_per_epoch = args.learning_steps_per_epoch
replay_memory_size = args.replay_memory_size

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
# resolution = (30, 45)
resolution = (100, 150)
episodes_to_watch = 10

model_savefile = args.save_dir + "/model-doom.pth"
best_model_savefile = args.save_dir + "/best_model-doom.pth"
save_model = True
load_model = args.test
skip_learning = not args.train

# Configuration file path

to_join = 'scenarios/' + str(args.env)  # 'simpler_basic'
config_file_path = os.path.join(os.path.dirname(vizdoom.__file__), to_join)

# Converts and down-samples the input image
def preprocess(img):
    # print(img(0,0,0])
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

def get_count(input_dim, kernel, stride):
    return (int((input_dim - kernel)/stride) + 1)

class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, 0, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


class Net(nn.Module):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        # print('*************************');
        features_x = get_count(get_count(resolution[0],6,3),3,2)
        features_y = get_count(get_count(resolution[1],6,3),3,2)
        self.features = 8 * features_x * features_y
        # print(features)
        self.fc1 = nn.Linear(self.features, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

criterion = nn.MSELoss()


def learn(s1, target_q):
    s1 = torch.from_numpy(s1)
    target_q = torch.from_numpy(target_q)
    s1, target_q = Variable(s1.to(device)), Variable(target_q.to(device))
    output = model(s1)
    loss = criterion(output, target_q)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def get_q_values(state):
    state = torch.from_numpy(state)
    state = Variable(state.to(device))
    return model(state)

def get_best_action(state):
    q = get_q_values(state)
    m, index = torch.max(q, 1)
    action = index.data.cpu().numpy()[0]
    return action


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q = get_q_values(s2).cpu().data.numpy()
        q2 = np.max(q, axis=1)
        target_q = get_q_values(s1).cpu().data.numpy()
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    max_x = max(x)
    x = [i - max_x for i in x]
    # print(max_x)
    to_return = np.exp(x) / np.sum(np.exp(x), axis=0)
    # print(x)
    # print(to_return)
    return to_return
    # return ( x / np.sum(x, axis=0) )

def calculate_entropy(action_values):
    action_values = action_values.data.cpu().numpy()[0]
    base = np.shape(action_values)[0]
    # print(action_values)
    action_probabilities = softmax(action_values)
    entropy = sc.entropy(action_probabilities, base = base)
    entropy = np.power(entropy,args.power)
    return entropy


# my_global_step = 0

def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""
    global my_global_step
    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        if args.exponential:
            return 0.99 ** epoch

        start_eps = 1.0
        end_eps = 0.01
        const_eps_epochs = args.max_eps_upto * epochs  # 10% of learning time
        eps_decay_epochs = args.decay_eps_upto * epochs  # 60% of learning time
        # return ( start_eps - ((start_eps - end_eps) / epochs) * epoch )
        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps
        return ( start_eps - ((start_eps - end_eps) / epochs) * epoch )

    s1 = preprocess(game.get_state().screen_buffer)
    # With probability eps make a random action.
    if args.bz:
        q_values = get_q_values(s1.reshape([1, 1, resolution[0], resolution[1]]))
        q_values = q_values.data.cpu().numpy()[0]
        temperature = exploration_rate(epoch)
        q_values = q_values/temperature
        a = np.random.choice(len(actions), p = softmax(q_values))      # boltzmann exploration
    else:
        if args.entropy:
            eps = calculate_entropy(get_q_values(s1.reshape([1, 1, resolution[0], resolution[1]])))
            if args.clip:
                if eps < args.clip_value:
                    eps = 0.01
                    # print('clipping happended')
                    my_global_step += 1
        else:
            eps = exploration_rate(epoch)

        if random() <= eps:
            a = randint(0, len(actions) - 1)
        else:
            # Choose the best action according to the network.
            s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
            a = get_best_action(s1)
        
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path + '.cfg')
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_doom_scenario_path(config_file_path + '.wad')
    game.set_episode_timeout(2100)
    game.init()
    print("Doom initialized.")
    return game

def one_hot_encoding(num_labels=None):
    actions = np.arange(num_labels)
    #empty one-hot matrix
    ohm = np.zeros((actions.shape[0], num_labels))
    ohm[np.arange(actions.shape[0]), actions] = 1
    return ohm.tolist()

if __name__ == '__main__':
    # Create Doom instance

    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    # actions = [list(a) for a in it.product([0, 1], repeat=n)]
    actions = one_hot_encoding(n)

    print('******** Number of Actions : ***************')
    print(len(actions))

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    if load_model:
        print("Loading model from: ", model_savefile)
        model = torch.load(model_savefile, map_location= device).to(device)
    else:
        model = Net(len(actions)).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    # if args.train:
    print("Starting the training!")
    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()

            my_global_step = 0
            
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                perform_learning_step(epoch)
                # print(my_global_step)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)
            print("%d percent clipped." % ((my_global_step / args.learning_steps_per_epoch)*100))

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            tensorboard.log_scalar('training_rewards', train_scores.mean(), epoch)

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    state = state.reshape([1, 1, resolution[0], resolution[1]])
                    best_action_index = get_best_action(state)

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f +/- %.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            tensorboard.log_scalar('test_rewards', test_scores.mean(), epoch)

            print("Saving the network weigths to:", model_savefile)
            torch.save(model, model_savefile)

            if test_scores.mean() > BEST_SCORE:
                BEST_SCORE = test_scores.mean()
                print("Saving the BEST network weigths to:", best_model_savefile)
                torch.save(model, best_model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    # game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    test_episodes_scores = []
    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            ENTROPY_STATE = calculate_entropy(get_q_values(state))
            print(ENTROPY_STATE)
            input("Press Enter to continue...")
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        score = game.get_total_reward()
        print("Total score: ", score)
        test_episodes_scores.append(score)
    test_episodes_scores = np.array(test_episodes_scores)
    print("Test Results: mean: %.1f +/- %.1f," % (
                test_episodes_scores.mean(), test_episodes_scores.std()), "min: %.1f" % test_episodes_scores.min(),
                  "max: %.1f" % test_episodes_scores.max())
