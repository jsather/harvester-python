""" Configuration options for the ddpg module.
"""
import os

# Learning parameters
buffer_size = 100000
batch_size = 16
pretrain_steps = 10000

# Experiment parameters
np_seed = 0
tf_seed = 0
max_episodes = 50000
max_episode_len = 1000
save_freq = 100

# Files and directories

try:
    harvester_python = [pp for pp in os.environ['PYTHONPATH'].split(":")
                        if 'harvester-python' in pp][0]
except IndexError as e:
    raise IndexError(
        'Could not find harvester-python workspace.' +
        ' Did you remember to update your PYTHONPATH?')
weights_file = ''
vars_file = ''
logfile = ''  # '/mnt/storage/logs/ddpg.log'
results_dir = os.path.join(harvester_python, 'storage', 'results')
buffer_dir = os.path.join(harvester_python, 'storage', 'buffer')) # '/mnt/storage/buffer'

# Other
headless = False

# GPU config
# device = '/GPU:0'
# gpu_usage = 0.8

# CPU config
device = ''
gpu_usage = 0.

# Embedding network
hidden_1_size_embedding = 400
hidden_2_size_embedding = 300
out_init_mag_embedding = 3e-3

# Actor and critic networks
actor_lr = 1e-4
critic_lr = 1e-3
critic_l2_scale = 1e-2
gamma = 0.99
tau = 1e-3

# Convolutional layers
filters_per_layer = [32, 32, 32, 32, 32]
stride_per_layer = [2, 2, 2, 2, 2]
kernel_size_per_layer = [3, 3, 3, 3, 3]
hidden_1_size_conv = 200
hidden_2_size_conv = 200
out_init_mag_conv = 3e-4
