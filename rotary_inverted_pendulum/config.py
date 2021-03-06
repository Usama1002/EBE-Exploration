import torch

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 10000
goal_score = 200
log_interval = 10
update_target = 1000
replay_memory_capacity = 50000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

small_epsilon = 0.0001
alpha = 0.5
beta_start = 0.1
