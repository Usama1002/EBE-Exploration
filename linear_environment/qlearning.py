
import numpy as np
import scipy.stats as sc

UCB_factor = 1.0

INIT_TEMP = 0.8
FINAL_TEMP = 0.1

# 0.8 - 0.2 is good

# INIT_TEMP = 0.9
# FINAL_TEMP = 0.1

TOTAL_STATES = 21

def calculate_entropy(action_values):
    base = np.shape(action_values)[0]
    max_action_value = max(action_values)
    action_values = [i - max_action_value for i in action_values]
    action_probabilities = np.exp(action_values) / (np.sum(np.exp(action_values)))
    entropy = sc.entropy(action_probabilities, base = base)
    return np.power(entropy,1)

def boltzmann_probs(action_values):
    return np.exp(action_values) / np.sum(np.exp(action_values))

# train using Q-learning
def Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon_profile, USE_ENTROPY=False, USE_BZ=False, USE_UCB=False):
    correct_Q_tablle = [[0.,0.],[gamma**0 , gamma**2],[gamma**1, gamma**3],[gamma**2, gamma**4],
    [gamma**3, gamma**5],[gamma**4, gamma**6],[gamma**5, gamma**7],[gamma**6, gamma**8],
    [gamma**7, gamma**9],[gamma**8, gamma**10],[gamma**9, gamma**9],[gamma**10, gamma**8],
    [gamma**9, gamma**7],[gamma**8, gamma**6],[gamma**7 , gamma**5],[gamma**6 , gamma**4],
    [gamma**5 , gamma**3],[gamma**4 , gamma**2],[gamma**3 , gamma**1],[gamma**2 , gamma**0],[0. , 0.] ]
    correct_Q_tablle  =np.array(correct_Q_tablle)
    
    mse = []
    Q = np.zeros([env.n_states, env.n_actions])
    action_visitation_counts = np.zeros([env.n_states, env.n_actions]) + 1
    state_visitation_count = np.zeros(TOTAL_STATES)

    avg_entropy_per_episode = []
    test_rewards = []

    n_steps = np.zeros(n_episodes) + max_steps
    sum_rewards = np.zeros(n_episodes)  # total reward for each episode
    epsilon = epsilon_profile.init
    temperature = INIT_TEMP
    for k in range(n_episodes):
        s = env.init_state
        entropys_for_all_states = []
        for i in range(TOTAL_STATES):
            entropys_for_all_states.append(calculate_entropy(Q[i]))
        entropys_for_all_states = np.array(entropys_for_all_states)
        avg_entropy_per_episode.append(entropys_for_all_states.mean())
        for j in range(max_steps):
            state_visitation_count[s] += 1
            epsilon_use = epsilon
            if USE_ENTROPY:
                epsilon_use = calculate_entropy(Q[s])
            if np.random.rand() < epsilon_use:
                a = np.random.randint(env.n_actions)      # random action
            else:
                mx = np.max(Q[s])
                a = np.random.choice(np.where(Q[s]==mx)[0])     # greedy action with random tie break
            if USE_BZ:
                # print(temperature)
                a = np.random.choice(env.n_actions, p = boltzmann_probs(Q[s]/temperature))      # boltzmann exploration
            if USE_UCB:
                qvalues_plus_visitation = Q + UCB_factor * np.sqrt(2*np.log(k+1) / action_visitation_counts)
                mx = np.max(qvalues_plus_visitation[s])
                # print(mx)
                a = np.random.choice(np.where(qvalues_plus_visitation[s]==mx)[0])

                action_visitation_counts[s,a] += 1

            sn = env.next_state[s,a]        
            r = env.reward[s,a]
            sum_rewards[k] += r
            Q[s,a] = (1.-alpha)*Q[s,a]+alpha*(r+gamma*np.max(Q[sn]))
            if env.terminal[sn]:
                n_steps[k] = j+1  # number of steps taken
                break
            s = sn
            epsilon = max(epsilon - epsilon_profile.dec_step, epsilon_profile.final)
        # mse.append(np.square(correct_Q_tablle - Q).mean(axis=None))
        mse.append(np.sum(np.square(correct_Q_tablle - Q)))
        epsilon = max(epsilon - epsilon_profile.dec_episode, epsilon_profile.final)
        temperature = max(temperature - epsilon_profile.dec_episode, FINAL_TEMP)
        _, test_episode_reward, _,_,_,_,_ = Q_test(Q, env, 1, 10, 0.01)
        test_rewards.append(test_episode_reward)
        # print(epsilon)
    return Q, n_steps, sum_rewards, mse, entropys_for_all_states, test_rewards, state_visitation_count


# run tests using action-value function table Q assuming epsilon greedy
def Q_test(Q, env, n_episodes, max_steps, epsilon):
    n_steps = np.zeros(n_episodes) + max_steps  # number of steps taken for each episode
    sum_rewards = np.zeros(n_episodes)          # total rewards obtained for each episode
    state = np.zeros([n_episodes, max_steps], dtype=np.int)      
    action = np.zeros([n_episodes, max_steps], dtype=np.int)      
    next_state = np.zeros([n_episodes, max_steps], dtype=np.int)
    reward = np.zeros([n_episodes, max_steps])

    avg_entropy_per_episode = []

    for k in range(n_episodes):
        entropys_for_all_states = []
        for i in range(TOTAL_STATES):
            entropys_for_all_states.append(calculate_entropy(Q[i]))
        entropys_for_all_states = np.array(entropys_for_all_states)
        avg_entropy_per_episode.append(entropys_for_all_states.mean())
        s = env.init_state
        for j in range(max_steps):
            state[k,j] = s
            if np.random.rand() < epsilon:
                a = np.random.randint(env.n_actions)      # random action
            else:
                mx = np.max(Q[s])
                a = np.random.choice(np.where(Q[s]==mx)[0])     # greedy action with random tie break
            action[k,j] = a
            sn = env.next_state[s,a]
            r = env.reward[s,a]
            next_state[k,j] = sn
            reward[k,j] = r
            sum_rewards[k] += r
            if env.terminal[sn]:
                n_steps[k] = j+1
                break
            s = sn
    return n_steps, sum_rewards, state, action, next_state, reward, avg_entropy_per_episode

