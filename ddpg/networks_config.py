""" networks_config.py contains configuration options for construction of the
    actor, critic, and embedding networks seen in networks.py
    Author: Jonathon Sather
    Last updated: 9/18/2018
"""
# GPU options
tf_cfg = dict({
    'allow_soft_placement': True,
    'log_device_placement': False})
device = '/GPU:0' #/device:GPU:0'

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
