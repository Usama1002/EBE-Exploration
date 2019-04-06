import random
import scipy.stats as sc
import tensorflow as tf
import numpy as np
from breakout_env import *
from neural_network import *
from wait import *
from collections import deque, namedtuple
# import tensorboard
import argparse
parser = argparse.ArgumentParser(description='Entropy Based Exploration')

random.seed(0)
np.random.seed(0)

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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    max_x = max(x)
    x = x - max_x
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    # return ( x / np.sum(x, axis=0) )

def calculate_entropy(action_values):
    base = np.shape(action_values)[0]
    action_probabilities = softmax(action_values)
    entropy = sc.entropy(action_probabilities, base = base)
    entropy = np.power(entropy,11)
    return entropy
    # if entropy <= 0.98 :
    #     return 0.001
    # else:
    #     return np.power(entropy,1)

def stretching(x):
    mean = np.mean(x)
    x = [i-mean for i in x]
    return [((np.exp(i) - 1) if i>=0 else -(np.exp(-i)-1)) for i in x]   

# This functions implements the DQN algorithm
def DQN_Implementation(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    max_steps,
                    replay_memory_size,
                    replay_memory_init_size,
                    update_target_estimator_every,
                    discount_factor,
                    batch_size,
                    epsilon_init,
                    epsilon_final,
                    USE_ENTROPY= False,
                    use_boltzmann = False,
                    SAVE_DATA= False,
                    saver= None,
                    save_path= None
                    ):

    saver=tf.train.Saver()

    epsilon_dec_step =1./max_steps
    epsilon_dec_step=0
    epsilon_profile_dec_episode=1./num_episodes
    # epsilon_profile_dec_episode=0

    # this tuple (data container) defines the structure of out stored transitions as said in algorithm where 'done'
    #  indicates if the state is terminal.
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    # replay memory
    replay_memory = []

    tensorboard = Tensorboard(save_path + "tensorboard")

    # Initializing the replay memory. We take random actions and store the transitions.
    print("Populating replay memory...")
    state = env.reset() # initializing the environment
    epsilon_curr=epsilon_init

    # initializing the replay memmory
    for i in range(replay_memory_init_size):
        action = np.random.randint(env.na) # taking random actions
        next_state, reward, done, _, _, _, _, _, _, _, _ = env.run(action-1) # acting on the environment
        # append new experience to the existing reply_memory
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done: # if terminal state is reached, then reset the environment.
            state = env.reset()
        else: # otherwise, proceed to the next state
            state = next_state

    total_rewards=np.zeros(num_episodes) # this vector stores the rewards for all episodes
    epsilon_curr=epsilon_init
    total_number_of_steps = 0 # this variable is helpful to decide when to update the target_estimator

    best_test_episode_steps = max_steps

    # Running the episodes
    for i_episode in range(num_episodes):
        greedy_actions=0;   first_action_greedy=0; last_action_random=0

        episode_reward=0 # initializing the current episode reward
        state = env.reset() #reset the environment
        epsilon_curr_step = epsilon_curr

        for t in range(max_steps):

            # here, we decide whether to update the target_estimator
            if total_number_of_steps>0 and total_number_of_steps % update_target_estimator_every == 0:
                sess.run(target_estimator.copy_parameters(q_estimator)) # updating the target_estimator
                total_number_of_steps=0
                print('Parameter Copied')

            if USE_ENTROPY:
                action_values = q_estimator.predict(sess, np.expand_dims(state,axis=0))[0]
                current_entropy = calculate_entropy(action_values)
                epsilon_curr_step = current_entropy
                

            # deciding the action based on epsilon-greedy policy
            if np.random.rand() < epsilon_curr_step:
                action = np.random.randint(env.na)  # random action
                if t==0:
                    first_action_greedy=1
                last_action_random = 1
            else:
                # taking predictions from q_estimator
                q_values = q_estimator.predict(sess, np.expand_dims(state,axis=0))
                greedy_actions+=1
                action = np.random.choice(np.where(q_values[0] == np.max(q_values))[0])

            if use_boltzmann:
                a = np.random.choice(env.na, p = softmax(action_values/epsilon_curr_step))

            # acting on the environment
            next_state, reward, done, _, _, _, _, _, _, _, _ = env.run(action -1)

            # If our replay memory is full, pop the first element before appending another
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save experience/transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))

            # update the current episode reward
            episode_reward+=reward

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # taking the next state estimated q_values from the target network
            q_values_next = target_estimator.predict(sess, next_states_batch)

            # making the target batch to be used for optimization as said in the algorithm
            targets_batch=np.zeros(batch_size)
            for i in range(batch_size):
                if done_batch[i]==0:
                    targets_batch[i]=reward_batch[i]+discount_factor*np.max(q_values_next[i])
                else:
                    targets_batch[i]=reward_batch[i]

            # updating the q_estimator (running optimization algorithm)
            q_estimator.update(sess, states_batch, action_batch, targets_batch)

            total_number_of_steps +=1

            if done: # if terminal state is reached, break the episode.
                break

            state = next_state

            last_action_random = 0
            
            tensorboard.log_scalar('exploration-fraction', epsilon_curr_step, total_number_of_steps)

            epsilon_curr_step=max(epsilon_curr_step - epsilon_dec_step, epsilon_final)

        if SAVE_DATA:
        	tensorboard.log_scalar('train_episode_rewards', episode_reward, i_episode)

        epsilon_curr = max(epsilon_curr - epsilon_profile_dec_episode, epsilon_final)

        # print ('Episode Number: %.f , Steps: %.f , Reward: %.2f Epsilon= %.3f, Greedy Actions=%.f, Last Random=%.f' % (
        #         i_episode + 1, t + 1, episode_reward, epsilon_curr, greedy_actions, last_action_random))

        # if USE_ENTROPY:
        #     print ('Episode Number: %.f , Steps: %.f , Reward: %.2f, Entropy: %.2f' % (
        #         i_episode + 1, t + 1, episode_reward, epsilon_curr_step))
        # else:
        #     print ('Episode Number: %.f , Steps: %.f , Reward: %.2f, Epsilon: %.2f' % (
        #         i_episode + 1, t + 1, episode_reward, epsilon_curr_step))

        # playing a test episode
        for i in range(1):
            test_episode_reward = 0 # initializing the current episode reward
            state = env.reset()
            total_number_of_test_steps = 0
            for t in range(max_steps):
                q_values = q_estimator.predict(sess, np.expand_dims(state,axis=0))
                greedy_actions+=1
                action = np.random.choice(np.where(q_values[0] == np.max(q_values))[0])
                next_state, reward, done, _, _, _, _, _, _, _, _ = env.run(action -1)
                test_episode_reward+=reward
                total_number_of_test_steps +=1
                state = next_state
                if done: # if terminal state is reached, break the episode.
                    break

        if test_episode_reward == 15 and total_number_of_test_steps < best_test_episode_steps:
            best_test_episode_steps = total_number_of_test_steps
            saver.save(sess=sess, save_path=save_path + 'best_model.ckpt')

        print('Episode Number: %.f, Test Reward: %.f' %(i_episode+1, test_episode_reward))

        if SAVE_DATA:
        	tensorboard.log_scalar('test_episode_reward', test_episode_reward, i_episode)

        if SAVE_DATA and ((i_episode+1)==num_episodes * 20 * 0.01 or (i_episode+1)==num_episodes * 40 * 0.01
                          or (i_episode+1)==num_episodes * 60 * 0.01 or (i_episode+1)==num_episodes * 80 * 0.01
                          or (i_episode+1)==num_episodes * 100 * 0.01):
           saver.save(sess=sess, save_path=save_path+'model.ckpt', global_step=i_episode+1)

    # gathering the episode rewards
    total_rewards[i_episode] = episode_reward



# *************************************************************************************************
#                       ------------------------- MAIN ----------------------------------
# *************************************************************************************************

# managing the arguments
parser.add_argument('--use_entropy', action='store_true', help='whether to use entropy')
parser.add_argument('--boltzmann', action='store_true', help='whether to use boltzmann')
parser.add_argument('--save', action='store_true', help='whether to save things')
parser.add_argument('--save_path', type=str, default='./data/unspecified_path/')
parser.add_argument('--episodes', type=int, default=2500)
args = parser.parse_args()

# initialize the environment
env = breakout_environment(5, 8, 3, 1, 2)
sess = tf.InteractiveSession()

# Creating  estimators
q_estimator = Estimator()
target_estimator = Estimator()

saver = tf.train.Saver(max_to_keep= 10)

USE_ENTROPY = args.use_entropy
SAVE_DATA = args.save
save_path = args.save_path

with tf.Session() as sess:
    # initializing tensorflow variables
    sess.run(tf.global_variables_initializer())

    print('*******************************************')
    print('Using Entropy' if USE_ENTROPY else 'NOT Using Entropy')
    print('*******************************************')

    # Calling the function that implements DQN training algorithm
    # Calling this function also defines all the hyper parameters used in this code
    DQN_Implementation(sess=sess,
                    env=env,
                    q_estimator=q_estimator,
                    target_estimator=target_estimator,
                    num_episodes=args.episodes,
                    max_steps=200,
                    replay_memory_size=1000,
                    replay_memory_init_size=100,
                    update_target_estimator_every=100,
                    discount_factor=0.95,
                    batch_size=10,
                    epsilon_init=1.0 ,
                    epsilon_final=0.01 ,
                    USE_ENTROPY = USE_ENTROPY,
                    use_boltzmann = boltzmann,
                    SAVE_DATA= SAVE_DATA,
                    saver = saver,
                    save_path= save_path
                    )

# _____________________ END of FILE ______________________________________________