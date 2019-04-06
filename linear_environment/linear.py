import numpy as np
import matplotlib.pyplot as plt
import pylab
from qlearning import *

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

# an instance of the environment
env = linear_environment()

class epsilon_profile: pass
epsilon = epsilon_profile()
epsilon.init = 1.0   # initial epsilon in e-greedy
epsilon.final = 0.01  # final epsilon in e-greedy
epsilon.dec_episode = 1. / n_episodes  # amount of decrement in each episode
# epsilon.dec_episode = 0.  # Uncomment it only when epsilon=1 is required during whole training.
epsilon.dec_step = 0.                  # amount of decrement in each step


NUMBER_OF_RUNS = 5

Q_entropy, n_steps_entropy, sum_rewards_entropy, mse_error_entropy, avg_ent_entropy, test_rewards_entropy, state_visit_entropy = [],[],[],[],[],[],[]
Q, n_steps, sum_rewards, mse_error, avg_ent, test_rewards, state_visit = [],[],[],[],[],[],[]
Q_bz, n_steps_bz, sum_rewards_bz, mse_error_bz, avg_ent_bz, test_rewards_bz, state_visit_bz = [],[],[],[],[],[],[]

for i in range(NUMBER_OF_RUNS):
	Q_run, n_run, sum_rewards_run, mse_error_run, avg_ent_run, test_rewards_run, state_visit_run = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon, USE_ENTROPY=True, USE_BZ=False, USE_UCB=False)
	Q_entropy.append(Q_run);
	n_steps_entropy.append(n_run);
	sum_rewards_entropy.append(sum_rewards_run);
	mse_error_entropy.append(mse_error_run);
	avg_ent_entropy.append(avg_ent_run);
	test_rewards_entropy.append(test_rewards_run)
	state_visit_entropy.append(state_visit_run)

for i in range(NUMBER_OF_RUNS):
	Q_run, n_run, sum_rewards_run, mse_error_run, avg_ent_run, test_rewards_run, state_visit_run = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon, USE_ENTROPY=False, USE_BZ=False, USE_UCB=False)
	Q.append(Q_run);
	n_steps.append(n_run);
	sum_rewards.append(sum_rewards_run);
	mse_error.append(mse_error_run);
	avg_ent.append(avg_ent_run);
	test_rewards.append(test_rewards_run)
	state_visit.append(state_visit_run)

for i in range(NUMBER_OF_RUNS):
	Q_run, n_run, sum_rewards_run, mse_error_run, avg_ent_run, test_rewards_run, state_visit_run = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon, USE_ENTROPY=False, USE_BZ=True, USE_UCB=False)
	Q_bz.append(Q_run);
	n_steps_bz.append(n_run);
	sum_rewards_bz.append(sum_rewards_run);
	mse_error_bz.append(mse_error_run);
	avg_ent_bz.append(avg_ent_run);
	test_rewards_bz.append(test_rewards_run)
	state_visit_bz.append(state_visit_run)

mse_error_entropy = np.mean(np.asarray(mse_error_entropy) , axis = 0)
mse_error = np.mean(np.asarray(mse_error) , axis = 0)
mse_error_bz = np.mean(np.asarray(mse_error_bz) , axis = 0)

plot_entropy = mse_error_entropy
plot_egreedy = mse_error
plot_bz      = mse_error_bz

plt.plot(plot_entropy, 'c', label='EBE', linewidth=3)
plt.plot(plot_egreedy, 'r', label='$\epsilon$-greedy', linewidth=3)
plt.plot(plot_bz, 'g', label='Boltzmann', linewidth=3)
plt.legend(fontsize=18)
plt.xlabel('number of training episodes', fontsize=18)
plt.ylabel('squared error', fontsize=18)
pylab.savefig('./squared_error.png',)

plt.show()

n_steps_entropy = np.array(n_steps_entropy)
# print('egreedy: ', str(state_visit.sum()))
# print('entropy: ', str(state_visit_entropy.sum()))
# print('entropy_total_steps: ', str(n_steps_entropy.sum()))
# print('boltzmann: ', str(state_visit_bz.sum()))
