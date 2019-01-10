""" config.py contains the configuration options for the ddpg subpackage.
"""
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
weights_file = ''
vars_file = ''
logfile = '' #'/mnt/storage/logs/ddpg.log' 
results_dir = '/mnt/storage/results'
buffer_dir = '/mnt/storage/buffer'

# Other
headless = True

# GPU options
tf_cfg = {
    'allow_soft_placement': True,
    'log_device_placement': False}
device = '/GPU:0'

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

