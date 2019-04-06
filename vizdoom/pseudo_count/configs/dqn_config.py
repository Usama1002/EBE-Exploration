"""
Config file for DQN on Atari 2600 suite
"""


class Config():
    # output config
    output_path = "results/dqn_privateeye/"
    # output_path = "/Users/kristyc/Downloads/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "test_privateeye.txt"

    # environment
    #env_name = "MontezumaRevengeNoFrameskip-v4"
    #env_name = "PongNoFrameskip-v4"
    env_name = "PrivateEyeNoFrameskip-v4"
    deep = True
    clip_grad = True

    # model and train config
    log_freq = 10000
    # save_freq = 5000

    # hyperparams
    batch_size = 32
    replay_buffer_size = 1000000
    frame_history_len = 4
    target_update_freq = 10000
    gamma = 0.99
    learning_freq = 4

    # others
    max_timesteps = 60000000
    learning_starts = 15000
    beta = 0.2  # MMC

    # optimizer
    learning_rate = 0.00025
    momentum = 0.95
    epsilon = 1e-2

    # epsilon-greedy
    egreedy_exploration = True
    mmc = True

    # bonus 
    bonus = True
    gaussian_ts = False
    alpha = 1.
