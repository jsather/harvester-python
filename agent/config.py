""" agent_config.py contains configuration options for AgentROS and PlantROS
    classes.
"""
import imp
import os
import pickle

import numpy as np

project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
global_config = imp.load_source('config',
    os.path.join(project_dir, 'config.py'))

# General
model_name = 'j2s6s200'
num_joints = 6
num_fingers = 2
finger_names = ['j2s6s200_joint_finger_1', 'j2s6s200_joint_finger_2']
joint_angles_init = np.array([-1.47, 3.00, 3.00, -0.09, 1.87, -2.23])
joint_limits = np.array([
    [-np.pi, np.pi],
    [30.0/180.0*np.pi, 330.0/180.0*np.pi],
    [30.0/180.0*np.pi, 330.0/180.0*np.pi],
    [-np.pi, np.pi],
    [30.0/180.0*np.pi, 330.0/180.0*np.pi],
    [-np.pi, np.pi]])
joint_names = [
    'j2s6s200_joint_1',
    'j2s6s200_joint_2',
    'j2s6s200_joint_3',
    'j2s6s200_joint_4',
    'j2s6s200_joint_5',
    'j2s6s200_joint_6']
joint_tolerance = 0.1 # If joint angles aren't within joint_tolerance of target, something is wrong.
camera_mount_offset =  np.array([[-0.051], [0], [-0.067]])

# Node
command_buffer_size = 5

# RL
obs_shape = (256, 256, 3)
existence_penalty = 0.1
invalid_goal_penalty = 1.0
detection_reward = 1.0
episode_measure = 'steps'
max_episode_time = 15
max_episode_steps = 100
plant_interval = 1
reward_threshold = 0.6

# Hemi stuff
with open(os.path.join(global_config.harvester_python, 'agent', 'lut')) as f:
    joints_data = pickle.load(f)
hemi_lut = joints_data['lut']
hemi_lut_thetas = joints_data['thetas']
hemi_lut_phis = joints_data['phis']
hemi_lut_mask = joints_data['valid_angles']
hemi_radius = 0.35
hemi_phi_max = np.pi/2
hemi_phi_min = 0.001
hemi_reset_angles = np.array([-np.pi/4, 0.1])
hemi_action_bound = np.array([0.05, 0.05])
hemi_action_penalty = 0.0 # 0.25
hemi_state_shape = [obs_shape, hemi_action_bound.shape]

# ROS launch
harvester_ros_path = os.path.join(global_config.catkin_ws, 'src',
    'harvester-ros')
harvester_default_cfg = {
    'harvester_robotName': 'harvester1',
    'kinova_robotName': model_name,
    'world_name': 'worlds/strawberry.world', #worlds/strawberry_camera.world
    'paused': True,
    'gui': True,
    'debug': False}

launch_world_delay = 17 # timed:15
load_robot_delay = 10 # timed:8
spawn_controllers_delay = 12 # timed:10
spawn_robot_delay = 17 # timed:15
move_home_delay = 15 # timed: 13
launch_moveit_delay = 17 # timed: 15
init_feed_delay = 20 # timed: ?
extra_delay = 32 # account for plant spawn + detector load
total_delay = launch_world_delay + load_robot_delay + \
    spawn_controllers_delay + spawn_robot_delay + move_home_delay + \
    launch_moveit_delay + init_feed_delay + extra_delay

# Ornstein Uhlenbeck process
mu = np.zeros(hemi_action_bound.shape)
sigma = 0.0075 # original 0.3 (for 40x larger action bound)
theta = 0.15

# Other
plant_model_dir = os.path.join(global_config.catkin_ws, 'harvester-sim',
    'harvester_gazebo', 'models', 'random_strawberry_plant')
feed_startup_script = os.path.join(global_config.harvester_python, 'image',
    'show_feed.py')
logfile = ''
