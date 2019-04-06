import numpy as np
import matplotlib.pyplot as plt
import pylab
from qlearning_count import *

def calculate_entropy(action_values):
    # print(action_values)
    base = np.shape(action_values)[0]
    max_action_value = max(action_values)
    action_values = [i - max_action_value for i in action_values]
    action_probabilities = np.exp(action_values) / (np.sum(np.exp(action_values)))
    entropy = sc.entropy(action_probabilities, base = 2)
    return np.power(entropy,1)

def smooth(scalars, weight=0.9):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

n_episodes = 200       # number of episodes to run
max_steps = 50      # max. # of steps to run in each episode
alpha = 0.2          # learning rate
gamma = 0.9          # discount factor
REWARD_STATE_0 = 1
REWARD_STATE_20 = 1

number_states = 21; number_actions = 2;

correct_Q_tablle = [[0.,0.],
                    [gamma**0 , gamma**2],
                    [gamma**1, gamma**3],
                    [gamma**2, gamma**4],
                    [gamma**3, gamma**5],
                    [gamma**4, gamma**6],
                    [gamma**5, gamma**7],
                    [gamma**6, gamma**8],
                    [gamma**7, gamma**9],
                    [gamma**8, gamma**10],

                    [gamma**9, gamma**9],

                    [gamma**10, gamma**8],
                    [gamma**9, gamma**7],
                    [gamma**8, gamma**6],
                    [gamma**7 , gamma**5],
                    [gamma**6 , gamma**4],
                    [gamma**5 , gamma**3],
                    [gamma**4 , gamma**2],
                    [gamma**3 , gamma**1],
                    [gamma**2 , gamma**0],
                    [0. , 0.]]

correct_Q_tablle  =np.array(correct_Q_tablle)

class linear_environment:
    def __init__(self):
        self.n_states = 21       # number of states
        self.n_actions = 2      # number of actions:2, 0- Left Move and 1- Right Move
        # self.next_state = np.array([[1,2],[1,1],[3,4],[3,3],[4,4]], dtype=np.int)    # next_state
        self.next_state = np.array([[0,0],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8,10],
                                    [9,11],[10,12],[11,13],[12,14],[13,15],[14,16],[15,17],[16,18],[17,19],
                                    [18,20],[20,20]], dtype=np.int)  # next_state
        self.reward = np.zeros([self.n_states,self.n_actions])            # reward for each (state,action)
        self.reward[self.n_states-1]=[0,0];    # Reward for state 20: 0: Left, 0: Right
        self.reward[self.n_states-2]=[0,REWARD_STATE_20];    # Reward for state 19: 0: Left, 1: Right
        self.reward[0] =[0,0];                 # Reward for state 0: 0: Left, 0: Right
        self.reward[1] = [REWARD_STATE_0, 0];               # Reward for state 1: 1: Left, 0: Right
        self.terminal = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], dtype=np.int)#1 if terminal, 0 otherwise
        self.init_state = 10     # initial state

env = linear_environment()

class epsilon_profile: pass
epsilon = epsilon_profile()
epsilon.init = 1.0   # initial epsilon in e-greedy
epsilon.final = 0.01  # final epsilon in e-greedy
epsilon.dec_episode = 1. / n_episodes  # amount of decrement in each episode
# epsilon.dec_episode = 0.  # Uncomment it only when epsilon=1 is required during whole training.
epsilon.dec_step = 0.                  # amount of decrement in each step


NUMBER_OF_RUNS = 10

# ENTROPY
Q, n_steps, sum_rewards, mse_error, avg_ent, test_rewards, state_visit = [],[],[],[],[],[],[]
for i in range(NUMBER_OF_RUNS):
    Q_run, n_run, sum_rewards_run, mse_error_run, avg_ent_run, test_rewards_run, state_visit_run = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon, USE_ENTROPY=True, USE_BZ=False, USE_UCB=False)
    Q.append(Q_run);
    n_steps.append(n_run);
    sum_rewards.append(sum_rewards_run);
    mse_error.append(mse_error_run);
    avg_ent.append(avg_ent_run);
    test_rewards.append(test_rewards_run)
    state_visit.append(state_visit_run)
mse_error_entropy = np.mean(np.asarray(mse_error) , axis = 0)

# UCB
Q, n_steps, sum_rewards, mse_error, avg_ent, test_rewards, state_visit = [],[],[],[],[],[],[]
for i in range(NUMBER_OF_RUNS):
    Q_run, n_run, sum_rewards_run, mse_error_run, avg_ent_run, test_rewards_run, state_visit_run = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon, USE_ENTROPY=False, USE_BZ=False, USE_UCB=True)
    Q.append(Q_run);
    n_steps.append(n_run);
    sum_rewards.append(sum_rewards_run);
    mse_error.append(mse_error_run);
    avg_ent.append(avg_ent_run);
    test_rewards.append(test_rewards_run)
    state_visit.append(state_visit_run)
mse_error_ucb = np.mean(np.asarray(mse_error) , axis = 0)

MBIE_factor = 1 # MBIE with beta=1
Q, n_steps, sum_rewards, mse_error, avg_ent, test_rewards, state_visit = [],[],[],[],[],[],[]
for i in range(NUMBER_OF_RUNS):
    Q_run, n_run, sum_rewards_run, mse_error_run, avg_ent_run, test_rewards_run, state_visit_run = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon, USE_MBIE = True , MBIE_beta = MBIE_factor)
    Q.append(Q_run);
    n_steps.append(n_run);
    sum_rewards.append(sum_rewards_run);
    mse_error.append(mse_error_run);
    avg_ent.append(avg_ent_run);
    test_rewards.append(test_rewards_run)
    state_visit.append(state_visit_run)
mse_error_mbie_1 = np.mean(np.asarray(mse_error) , axis = 0)

MBIE_factor = 5 # MBIE with beta=5
Q, n_steps, sum_rewards, mse_error, avg_ent, test_rewards, state_visit = [],[],[],[],[],[],[]
for i in range(NUMBER_OF_RUNS):
    Q_run, n_run, sum_rewards_run, mse_error_run, avg_ent_run, test_rewards_run, state_visit_run = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon, USE_MBIE = True , MBIE_beta = MBIE_factor)
    Q.append(Q_run);
    n_steps.append(n_run);
    sum_rewards.append(sum_rewards_run);
    mse_error.append(mse_error_run);
    avg_ent.append(avg_ent_run);
    test_rewards.append(test_rewards_run)
    state_visit.append(state_visit_run)
mse_error_mbie_5 = np.mean(np.asarray(mse_error) , axis = 0)

MBIE_factor = 100 # MBIE with beta=10
Q, n_steps, sum_rewards, mse_error, avg_ent, test_rewards, state_visit = [],[],[],[],[],[],[]
for i in range(NUMBER_OF_RUNS):
    Q_run, n_run, sum_rewards_run, mse_error_run, avg_ent_run, test_rewards_run, state_visit_run = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon, USE_MBIE = True , MBIE_beta = MBIE_factor)
    Q.append(Q_run);
    n_steps.append(n_run);
    sum_rewards.append(sum_rewards_run);
    mse_error.append(mse_error_run);
    avg_ent.append(avg_ent_run);
    test_rewards.append(test_rewards_run)
    state_visit.append(state_visit_run)
mse_error_mbie_10 = np.mean(np.asarray(mse_error) , axis = 0)

MBIE_factor = 250 # MBIE with beta=10
Q, n_steps, sum_rewards, mse_error, avg_ent, test_rewards, state_visit = [],[],[],[],[],[],[]
for i in range(NUMBER_OF_RUNS):
    Q_run, n_run, sum_rewards_run, mse_error_run, avg_ent_run, test_rewards_run, state_visit_run = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon, USE_MBIE = True , MBIE_beta = MBIE_factor)
    Q.append(Q_run);
    n_steps.append(n_run);
    sum_rewards.append(sum_rewards_run);
    mse_error.append(mse_error_run);
    avg_ent.append(avg_ent_run);
    test_rewards.append(test_rewards_run)
    state_visit.append(state_visit_run)
mse_error_mbie_250 = np.mean(np.asarray(mse_error) , axis = 0)

plot_entropy = mse_error_entropy
plot_ucb = mse_error_ucb
plot_mbie_1 = mse_error_mbie_1
plot_mbie_5 = mse_error_mbie_5
plot_mbie_10 = mse_error_mbie_10
plot_mbie_250 = mse_error_mbie_250

plt.plot(plot_entropy,label='EBE', linewidth=3)
plt.plot(plot_ucb,  label='UCB', linewidth=3)
plt.plot(plot_mbie_1, 'y--', label='MBIE-EB, beta=1', linewidth=3)
plt.plot(plot_mbie_5, 'k--', label='MBIE-EB, beta=5', linewidth=3)
plt.plot(plot_mbie_10, 'r--', label='MBIE-EB, beta=100', linewidth=3)

plt.legend(fontsize=15)
plt.xlabel('number of training episodes', fontsize=18)
plt.ylabel('squared error', fontsize=18)
pylab.savefig('./squared_error.png')


plt.show()