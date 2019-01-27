""" agent_ros.py contains interface to talk to the harvester arm in
    Gazebo/ROS.

    Structure inspired by:
    https://github.com/cbfinn/gps/tree/master/python/gps/agent/ros

    Author: Jonathon Sather
    Last updated: 10/02/2018

    # TODO: Get rid of methods don't think will ever use! Also get rid of "get_xx" methods... bad python style
""" 
from __future__ import print_function
import os
import sys
import time
from time import sleep
import signal
import subprocess
import copy

import numpy as np
import cv2
import rospy
import moveit_commander
import tf_conversions
from collections import deque
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import Bool
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import SwitchController
from controller_manager_msgs.srv import ListControllers
from dynamic_reconfigure.client import Client

import agent_utils
import config as agent_cfg
from plant_ros import PlantROS  

try:
    from image.image_ros import RewardROS
except ImportError as e:
    print('Unable to import RewardROS: ', e)

import pdb

class AgentROS(object):
    """ Agent superclass for interfacting with harvesting robot through
        ROS/Gazebo.
    """

    def __init__(self, headless=False):
        """ Initializes ROS/Gazebo agent. """
        if agent_cfg.logfile != '':
            sys.stdout = open(agent_cfg.logfile, 'a')
            sys.stderr = open(agent_cfg.logfile, 'a')

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        # Robot
        self.model_name = agent_cfg.model_name
        self.num_joints = agent_cfg.num_joints
        self.num_fingers = agent_cfg.num_fingers
        self.joint_limits = agent_cfg.joint_limits
        self.joint_tolerance = agent_cfg.joint_tolerance
        self.camera_mount_offset = agent_cfg.camera_mount_offset
        self.obs_shape = agent_cfg.obs_shape
        self.joint_angles = None
        self.obs = None

        # RL
        self.detection_reward = agent_cfg.detection_reward
        self.existence_penalty = agent_cfg.existence_penalty
        self.invalid_goal_penalty = agent_cfg.invalid_goal_penalty
        self.episode_measure = agent_cfg.episode_measure
        self.max_episode_time = agent_cfg.max_episode_time
        self.max_episode_steps = agent_cfg.max_episode_steps
        self.plant_interval = agent_cfg.plant_interval
        self.reward_threshold = agent_cfg.reward_threshold
        self.start_time = None
        self.step_count = 0
        self.episode_count = 0
        self.trajectory_fraction = 0
        self.last_return = None
        
        # Start simulation 
        rospy.init_node('harvester_agent_ros_node')
        self._start_gazebo(headless=headless)
        self.plant = PlantROS()

        self._init_controller_names()
        self._init_controllers()
        self._init_pubs_and_subs()
        self._init_moveit()
   
    def _init_controller_names(self):
        """ Initializes joint/finger names and trajectory controller names. """
        self.joint_names = []
        for i in range(self.num_joints):
            name = self.model_name + '_joint_' + str(i+1)
            self.joint_names.append(name)

        self.finger_names = []
        for i in range(self.num_fingers):
            name = self.model_name + '_joint_finger_' + str(i+1)
            self.finger_names.append(name)

        self.joint_trajectory_controller = 'effort_joint_trajectory_controller'
        self.finger_trajectory_controller = \
            'effort_finger_trajectory_controller'

    def _init_controllers(self, verbose=False):
        """ (Re)initializes controllers for robotic arm. """
        # Stop all controllers except joint_state_publisher
        self._stop_controllers(verbose=verbose)

        switch_loc = '/' + self.model_name + \
            '/controller_manager/switch_controller'
        rospy.wait_for_service(switch_loc)

        # Initialize joint and finger trajectory controllers
        if verbose:
            print("Initializing trajectory controllers... ")
            sys.stdout.flush()

        switched = False
        attempt = 0
        try:
            switch_controller = rospy.ServiceProxy(switch_loc,
                SwitchController)

            while not switched:
                attempt += 1
                ret = switch_controller([self.joint_trajectory_controller,
                    self.finger_trajectory_controller],
                    [], 2)
                if ret.ok:
                    switched = True

            if verbose:
                print("Done.")
                sys.stdout.flush()

        except rospy.ServiceException, e:
            print("\nService call failed: %s", e)

    def _init_pubs_and_subs(self, verbose=True):
        """ Initializes position and trajectory publishers and subscribers, and
            camera subscriber.
        """
        if verbose:
            print("Initializing ROS publishers and subscribers... ")
            sys.stdout.flush()

        # Joint position subscriber
        self.joint_position_sub = rospy.Subscriber(
            '/' + self.model_name + '/joint_states', JointState,
            self._process_joint_data, queue_size=1)

        # Joint position republisher
        self.joint_position_repub = rospy.Publisher('/joint_states',
            JointState, queue_size=1)

        # MoveIt trajectory publisher
        self.moveit_trajectory_pub = rospy.Publisher(
            '/move_group/display_planned_pamemberdatath',
            moveit_msgs.msg.DisplayTrajectory, queue_size=1)

        # Camera image subscriber
        self.camera_sub = rospy.Subscriber('/harvester/camera1/image_raw',
            Image, self._process_image_data, queue_size=1)

        # Step publisher
        self.step_pub = rospy.Publisher(
            '/' + self.model_name + '/step', String, queue_size=1)

        # State count publisher
        self.step_count_pub = rospy.Publisher(
            '/' + self.model_name + '/step_count', String, queue_size=1)

        if verbose:
            print("Done.")
            sys.stdout.flush()

    def _init_moveit(self, verbose=True):
        """ Initializes MoveIt interface for trajectory planning. """
        if verbose:
            print("Initializing MoveIt!... ")
            sys.stdout.flush()

        moveit_commander.roscpp_initialize(sys.argv) # wut
        self.moveit_robot = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_arm_group = moveit_commander.MoveGroupCommander('arm')
        self.moveit_gripper_group = moveit_commander.MoveGrhttps://www.google.com/search?client=ubuntu&channel=fs&q=matplotlib+name+figure&ie=utf-8&oe=utf-8oupCommander(
            'gripper')
        self.moveit_ik = agent_utils.GetIK(group='arm', verbose=False)
        self.moveit_config = Client('/move_group/trajectory_execution')
        self.moveit_config.update_configuration({'allowed_start_tolerance':0.0})

        if verbose:
            print("Done.")

    def _start_gazebo(self, verbose=True, headless=False):
        """ Starts Gazebo instance. """
        self.sim = agent_utils.HarvesterSimulation()
        self.sim.run(verbose=verbose, headless=headless)

    def _stop_gazebo(self):
        """ Stops Gazebo instance. """
        self.sim.kill()

    def _stop_controllers(self, verbose=False):
        """ Stops all running controllers, except Joint State Publisher. """
        # Make list of running controllers
        list_loc = '/' + self.model_name + \
            '/controller_manager/list_controllers'
        rospy.wait_for_service(list_loc)

        try:
            list_controllers = rospy.ServiceProxy(list_loc, ListControllers)
            controllers = list_controllers()
        except rospy.ServiceException, e:
            print("Service call failed: %s", e)

        running = []
        for ctr in controllers.controller:
            if (ctr.state == 'running' and
                ctr.name !=  'joint_state_controller'):
                running.append(ctr.name)

        # Stop running controllers
        switch_loc = '/' + self.model_name + \
            '/controller_manager/switch_controller'
        rospy.wait_for_service(switch_loc)

        if verbose:
            print("Stopping controllers... ")
            sys.stdout.flush()
        try:
            switch_controller = rospy.ServiceProxy(switch_loc, SwitchController)
            stopped = False
            attempt = 0
            while not stopped:
                attempt += 1
                ret = switch_controller([], running, 2)
                if ret.ok:
                    stopped = True
            if verbose:
                print("Done.")
                sys.stdout.flush()
        except rospy.ServiceException, e:
            print("Service call failed: %s", e)

    def _process_joint_data(self, ros_data):
        """ Callback function for joint topic. Receives JointState object as
            input and stores joint angle data.
        """
        self.joint_position_repub.publish(ros_data)
        all_angles = np.asarray(ros_data.position)
        self.joint_angles = all_angles[:-2] # Remove finger positions

    def _process_image_data(self, ros_data):
        """ Callback function for camera topic. Receives raw input and outputs
            Numpy array of pixel intensities.
        """
        flat = np.fromstring(ros_data.data, np.uint8)
        full = np.reshape(flat, (ros_data.height, ros_data.width, -1))
        self.obs = cv2.resize(full, (self.obs_shape[0], self.obs_shape[1]))

    def _publish_step(self, step):
        """ Publishes step to ROS topic. """
        step_data = String(str(step))
        self.step_pub.publish(step_data)

    def _publish_step_count(self):
        """ Publishes state count to ROS topic. """
        count_data = String(str(self.step_count))
        self.step_count_pub.publish(count_data)

    def exit_gracefully(self, sig, frame):
        """ Kill spawned processes when program terminated. """
        print('Signal: ' + str(sig) + '. Killing agent and related processes.')
        self._stop_gazebo()
        sys.exit()

    def get_ee_pose(self):
        """ Returns current end effector pose. """
        return self.moveit_arm_group.get_current_pose()

    def get_camera_pose(self):
        """ Returns current camera pose. """
        camera_pose = self.moveit_arm_group.get_current_pose()
        rotation_matrix = tf_conversions.transformations.quaternion_matrix(
            [camera_pose.pose.orientation.x,
            camera_pose.pose.orientation.y,
            camera_pose.pose.orientation.z,
            camera_pose.pose.orientation.w])

        mount_offset = np.vstack((self.camera_mount_offset, np.ones(1)))
        rel_offset = np.matmul(rotation_matrix, mount_offset)

        camera_pose.pose.position.x += rel_offset[0]
        camera_pose.pose.position.y += rel_offset[1]
        camera_pose.pose.position.z += rel_offset[2]
        return camera_pose

    def get_ik(self, pose, seed_angles=None, ik_attempts=0,
        avoid_collisions=True):
        """ Computes Moveit's /compute_ik service to get inverse kinematics for
            robot arm.
        """
        if seed_angles:
            seed = moveit_msgs.msg.RobotState()
            seed.joint_state.name = self.joint_names
            seed.joint_state.position = seed_angles
            if ik_attempts == 0:
                ik_attempts = 1
        else:
            seed=None

        return self.moveit_ik.get_ik(pose, seed=seed, ik_attempts=ik_attempts,
            avoid_collisions=avoid_collisions)

    def get_joint_limits(self):
        """ Returns joint limits. """
        return self.joint_limits

    def get_joint_angles(self):
        """ Fetches joint angles from joint_state_publisher topic. """
        return self.joint_angles

    def get_camera_image(self):
        """ Fetches latest camera image. """
        return self.obs

    def get_episode_count(self):
        """ Returns number of completed episodes in session. """
        return self.episode_count

    def get_last_return(self):
        """ Returns last recorded return from taking step in environment. """
        return self.last_return

    def stamp_pose(self, pose):
        """ Returns PoseStamped object with given pose and header. """
        ps = geometry_msgs.msg.PoseStamped()
        ps.header.frame_id = self.moveit_robot.get_planning_frame()
        ps.pose = pose
        return ps

    def start_episode(self):
        """ Starts timer for new episode. """
        self.start_time = time.time()
        self.step_count = 0

    def episode_finished(self):
        """ Checks if episode is finished."""
        if self.episode_measure == 'time':
            return time.time() - self.start_time > self.max_episode_time
        else:
            return self.step_count >= self.max_episode_steps

    def execute_joint_trajectory(self, goal, group='arm', wait=True):
        """ Executes trajectory to specified joint positions. """
        if group == 'arm':
            move_group = self.moveit_arm_group
        elif group == 'gripper':
            move_group = self.moveit_gripper_group
        else:
            print("Invalid move group specified. Cannot execute trajectory.")
            return False

        return move_group.go(goal, wait=wait)

    def execute_trajectory(self, goal=None, group='arm', name=None, wait=True):
        """ Executes trajectory to specified pose using Moveit. Note: Can only
            execute named trajectory for gripper group. Use
            'execute_joint_trajectory()' to explicitly set joint positions.
        """
        if group == 'arm':
            move_group = self.moveit_arm_group
            move_group.clear_pose_targets()

            if name:
                move_group.set_named_target(name)
            else:
                move_group.set_pose_target(goal)
        elif group == 'gripper':
            move_group = self.moveit_gripper_group
            move_group.clear_pose_targets()

            if name:
                move_group.set_named_target(name)
            else:
                print("Invalid group/name combination specified. Cannot " +
                    "execute trajectory")
                return False
        else:
            print("Invalid move group specified. Cannot execute trajectory.")
            return False

        move_group.plan()
        return move_group.go(wait=wait)

    def get_state_shape(self):
        """ Returns shape of state feedback quantities. """
        raise NotImplementedError("Must be implemented in subclass.")

    def get_action_shape(self):
        """ Returns action shape. """
        raise NotImplementedError("Must be implemented in subclass.")

    def get_action_bound(self):
        """ Returns action bound. """
        raise NotImplementedError("Must be implemented in subclass.")

    def get_state(self):
        """ Returns last state. """
        raise NotImplementedError("Must be implemented in subclass.")

    def get_reward(self):
        """ Reward at current state + info. """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """ Resets harvesting arm to starting position. """
        raise NotImplementedError("Must be implemented in subclass.")

    def step(self):
        """ Take action with harvesting arm and get updated observations.
        """
        raise NotImplementedError("Must be implemented in subclass.")

class FreeAgentROS(AgentROS):
    """ Agent class that uses joint trajectory control for taking steps in the
        environment.
        Note: This has never been implemented, so should do some trials and
            debugging before using.
    """

    def __init__(self, action_bound=None):
        """ Initializes ROS/Gazebo agent. """
        super(FreeAgentROS, self).__init__()

        if action_bound:
            self.action_bound = action_bound
        else: # Default to 0.01 times joint span
            span = self.joint_limits[:, 1] - self.joint_limits[:, 0]
            self.action_bound = 0.01*span

        self.reset()

    def get_state_shape(self):
        """ Returns [obs_shape, joints_shape]. """
        return [self.obs.shape, self.joint_angles.shape]

    def get_action_shape(self):
        """ Returns action shape. """
        return self.joint_angles.shape

    def get_action_bound(self):
        """ Returns action bound. """
        return self.action_bound

    def get_state(self):
        """ Returns last state. """
        return [self.obs, self.joint_angles]

    def get_reward(self):
        """ Returns reward at current state. """
        raise NotImplementedError(
            "FreeAgentROS under construction. Check back later.")

    def reset(self):
        """ Resets harvesting arm to starting position. """
        self.execute_trajectory(group='gripper', name='Open')
        self.execute_trajectory(name='Home') # Make sure this exists...

    def step(self, joint_inc):
        """ Take action with harvesting arm and get updated observations.
        """
        info = {}
        new_angles = self.get_joint_angles() + joint_inc
        info['move_msg'] = self.execute_joint_trajectory(goal=new_angles)
        obs = self.get_state()
        info['reward_msg'], reward = self.get_reward()
        done = self.episode_finished()

        return obs, reward, done, info

class HemiAgentROS(AgentROS):
    """ Agent class whos main action space is constrained to a hemispherical
        manifold above the strawberry plant.
    """

    def __init__(self, headless=False, feed=False, detector=True, 
        tf_session=None):
        """ Initializes ROS/Gazebo agent. """
        super(HemiAgentROS, self).__init__(headless=headless)

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        self.lut = agent_cfg.hemi_lut
        self.lut_thetas = agent_cfg.hemi_lut_thetas
        self.lut_phis = agent_cfg.hemi_lut_phis
        self.lut_mask = agent_cfg.hemi_lut_mask

        self.radius = agent_cfg.hemi_radius # m
        self.phi_max = agent_cfg.hemi_phi_max # rad
        self.phi_min = agent_cfg.hemi_phi_min # rad
        self.reset_angles = agent_cfg.hemi_reset_angles # rad
        self.action_bound = agent_cfg.hemi_action_bound
        self.action_penalty = agent_cfg.hemi_action_penalty
        
        self.reset(random=False, verbose=True, new_episode=True) 

        if feed:
            self._start_feed()

        if detector:
            self.detector_feedback = RewardROS(tf_session=tf_session)

    def _start_feed(self, verbose=True):
        """ Starts feed process. """
        if verbose:
            print("Setting up camera feed... ")
            sys.stdout.flush()

        self.feed = agent_utils.Feed()
        self.feed.run()

        if verbose:
            print("Done.")
            sys.stdout.flush()
    
    def _stop_feed(self):
        """ Stops feed process. """
        try:
            self.feed.kill()
        except Exception, e:
            print("Failed to kill feed (ignore if disabled): %s", e)

    def _angles_to_joints(self, theta, phi):
        """ Converts [theta, phi] position along hemisphere into joint angles
            using lut.
        """
        theta = theta % (2*np.pi)
        theta_idx = (np.abs(self.lut_thetas - theta)).argmin()
        phi_idx = (np.abs(self.lut_phis - phi)).argmin()

        if phi < 0.0 or phi > np.pi/2:
            valid = 0
        else:
            valid = self.lut_mask[theta_idx, phi_idx]

        return self.lut[theta_idx, phi_idx], valid
    
    def _angles_to_pose(self, theta, phi):
        """ Converts [theta, phi, rho] position along hemisphere into Pose
            object in world frame.
        """
        pose = geometry_msgs.msg.Pose()

        # Find position on hemisphere
        x = self.radius * np.sin(phi) * np.cos(theta)
        y = self.radius * np.sin(phi) * np.sin(theta)
        z = self.radius * np.cos(phi)
        position = geometry_msgs.msg.Point(x, y, z)

        # Find orientation directed at center
        to_ctr = -np.reshape([x, y, z], (3, 1))
        x_hat = np.reshape([1, 0, 0], (3, 1))
        z_hat = np.reshape([0, 0, 1], (3, 1))

        if theta < 0 and theta > -np.pi:
            r1 = np.arccos(np.dot(np.transpose(x_hat[0:2]), to_ctr[0:2]) /
                (np.linalg.norm(x_hat[0:2]) * np.linalg.norm(to_ctr[0:2])))
        else:
            r1 = -np.arccos(np.dot(np.transpose(x_hat[0:2]), to_ctr[0:2]) /
                (np.linalg.norm(x_hat[0:2]) * np.linalg.norm(to_ctr[0:2])))
        r2 = np.arccos(np.dot(np.transpose(z_hat), to_ctr) /
                (np.linalg.norm(z_hat) * np.linalg.norm(to_ctr)))

        r1_matrix = tf_conversions.transformations.rotation_matrix(r1,
            [0, 0, 1])
        r2_matrix = tf_conversions.transformations.rotation_matrix(r2,
            [0, 1, 0])
        r_matrix = np.matmul(r1_matrix, r2_matrix)
        orientation = geometry_msgs.msg.Quaternion(
            *tf_conversions.transformations.quaternion_from_matrix(r_matrix))

        pose.position = position
        pose.orientation = orientation

        return pose
    
    def _pose_to_angles(self, pose):
        """ Returns theta, phi angles corresponding to given pose. """
        theta = np.arctan2(pose.position.y, pose.position.x)
        phi = np.arctan2(np.sqrt(pose.position.x**2 + pose.position.y**2),
            pose.position.z)

        return theta, phi

    def _clamp_phi(self, phi):
        """ Method for clamping phi inside operating range. """
        return max(min(self.phi_max, phi), self.phi_min)
    
    def check_joint_error(self, tol=None):
        """ Raises exception if maximum joint error is not within specified
           margin.
        """
        if tol is None:
            tol = self.joint_tolerance

        joint_error = self.max_joint_error()
        if joint_error > tol:
            raise RuntimeError('Joint error of ' + str(joint_error) + 
                ' above threshold of ' + str(tol) + 
                ' Something is likely very, very wrong.')

    def exit_gracefully(self, sig, frame):
        """ Kill spawned processes when program terminated. """
        print('Signal: ' + str(sig) + '. Killing agent and related processes.')
        self._stop_feed()
        self._stop_gazebo()
        sys.exit()

    def get_state_shape(self):
        """ Returns [obs_shape, hemi_position_shape]. """
        return [self.obs.shape, self.action_bound.shape]

    def get_action_shape(self):
        """ Returns action shape. """
        return self.action_bound.shape

    def get_action_bound(self):
        """ Returns action bound. """
        return self.action_bound

    def get_cur_angles(self):
        """ Returns theta values corresponding to current position. """
        cur_pose = self.get_ee_pose()
        return self._pose_to_angles(cur_pose.pose)

    def get_state(self):
        """ Returns last state. """
        return [self.obs, self.get_cur_angles()]

    def get_reward(self, action, in_bounds):
        """ Returns reward at current state. """
        if not in_bounds:
            reward = -self.invalid_goal_penalty
        elif self.detector_feedback.get_reward() > self.reward_threshold:
            reward = self.detection_reward
        else:
            reward = -self.existence_penalty

        # detect_reward = self.detection_reward * \
        #     int(self.detector_feedback.get_reward() > self.reward_threshold)
        # act_norm = 1/self.action_bound * np.array(action)
        # act_penalty = self.action_penalty * np.inner(act_norm, act_norm)
        # goal_penalty = self.invalid_goal_penalty * (not in_bounds)
        
        # reward = detect_reward - self.existence_penalty - goal_penalty - \
        #     act_penalty
        return reward
    
    def get_last_return(self):
        """ Returns last return stored in memberdata. """
        return self.last_return 

    def max_joint_error(self):
        """ Gives distance in hemi-space to last commanded target. """
        current = np.array(self.moveit_arm_group.get_current_joint_values())
        current = current % (2*np.pi)
        target = np.array(self.moveit_arm_group.get_joint_value_target())
        target = target % (2*np.pi)

        errors = current - target 
        errors_adjusted = [min(abs(x), 2*np.pi - abs(x)) for x in errors]

        return np.linalg.norm(errors_adjusted, np.inf)

    def move_to_angles_lut(self, theta, phi, wait=True):
        """ Moves to specified joint angles in joint space using look up table.
        """
        cur_theta, cur_phi = self.get_cur_angles()
        if theta is None:
            theta = copy.copy(cur_theta)

        if phi is None:
            phi = copy.copy(cur_phi)
        
        self.latest_commanded_angles = (theta, phi)

        joint_angles, valid = self._angles_to_joints(theta, phi)

        if not valid: # Angles out of workspace.
            fraction = 0.0
        else:
            fraction = self.execute_joint_trajectory(joint_angles, wait=wait)
        
        return {'fraction': fraction, 'valid': valid}

    def move_to_angles_hemi(self, theta=None, phi=None, max_plan=5,
        execute_partial_plan=False):
        """ Moves arm along hemisphere to specific angles. If no angle
            specified, stays at current value.
        """
        # Determine travel directions
        cur_theta, cur_phi = self.get_cur_angles()
        if theta is None and phi is None:
            theta = copy.copy(cur_theta)
            phi = copy.copy(cur_phi)
        elif theta is None:
            theta = copy.copy(cur_theta)
        elif phi is None:
            phi = copy.copy(cur_phi)
        else:
            pass

        phi = self._clamp_phi(phi)
        self.latest_commanded_angles = (theta, phi)

        phi_dir = np.sign(phi - cur_phi)

        cur_theta_shifted = (cur_theta + np.pi/2) % (2*np.pi)
        theta_shifted = (theta + np.pi/2) % (2*np.pi)
        theta_dir = np.sign(theta_shifted - cur_theta_shifted)
        theta_dist = abs(theta_shifted - cur_theta_shifted)

        theta_steps = int(np.ceil(theta_dist / self.action_bound[0]))
        phi_steps = int(np.ceil(abs(phi - cur_phi) / self.action_bound[1]))

        steps = max(theta_steps, phi_steps)
        theta_inc = theta_dist / steps
        phi_inc = abs(phi - cur_phi) / steps

        # Determine waypoints
        waypoints = []

        for step in range(1, steps+1):
            wtheta = float(cur_theta + theta_dir*step*theta_inc)
            wtheta = (wtheta + np.pi) % (2*np.pi) - np.pi # Make sure not out of range

            wphi = float(cur_phi + phi_dir*step*phi_inc)
            wtarget = self._angles_to_pose(wtheta, wphi)
            waypoints.append(wtarget)

        # Plan trajectory and execute if appropriate
        num_plan = 0
        while True:
            (plan, fraction) = self.moveit_arm_group.compute_cartesian_path(
                waypoints, 0.01, 0.0)
            if fraction == 1.0:
                self.moveit_arm_group.execute(plan)
                break
            elif num_plan > max_plan:
                if execute_partial_plan:
                    self.moveit_arm_group.execute(plan)
                break
            else:
                num_plan += 1

        return {'fraction': fraction}

    def move_to_angles(self, theta=None, phi=None, max_plan=5,
        execute_partial_plan=False, along_hemi=False, wait=True):
        """ Moves to specified joint angles on hemisphere. Returns dictionary 
            with trajectory information.
        """
        if along_hemi: # NOTE: No "wait" option here (yet (maybe))
            return self.move_to_angles_hemi(theta=theta, phi=phi, max_plan=max_plan,
                execute_partial_plan=False) 
        else:
            return self.move_to_angles_lut(theta=theta, phi=phi, wait=wait)

    def move_to_random(self, wait=True):
        """ Moves to uniformly sampled joint angles on hemisphere. Uses look up
            table for consistency.
        """
        while True:
            theta = 2*np.pi*np.random.rand() - np.pi
            phi = np.pi/2*np.random.rand()
            info = self.move_to_angles(theta=theta, phi=phi, along_hemi=False,
                wait=wait)
            if info['valid']:
                return info['fraction']

        return False # Lol

    def on_hemi(self, tol=0.05):
        """ Returns boolean whether agent is (approximately) on hemisphere, or
            not. Useful for error checking. Currently only considers distance
            from center and not orientation.
        """
        pos = self.get_ee_pose().pose.position
        center_dist = np.linalg.norm(np.array([pos.x, pos.y, pos.z]))
        if abs(center_dist - self.radius) > tol:
            return False
        return True

    def reset(self, random=False, new_episode=True, along_hemi=False, 
        verbose=False, safe_mode=False, kill_on_fail=False, 
        plant_file=None):
        """ Resets harvesting arm to starting position. 
            TODO: Add documentation for args...
        """
        if verbose:
            print("Resetting arm... ")
            sys.stdout.flush()
        
        if new_episode:
            self.episode_count += 1
            self.start_episode()
            self._publish_step(None)

            if (not self.plant.is_spawned() or 
                self.episode_count % self.plant_interval == 0):

                if verbose: 
                    print('Spawning new plant...')
                    sys.stdout.flush()
                
                if plant_file is not None:
                    self.plant.new(plant_model_prefix=plant_file)
                else:
                    self.plant.new()

                sleep(15)
                if verbose:
                    print('Done.')
                    sys.stdout.flush()

        self.execute_trajectory(group='gripper', name='Open') # Not doing this on init? TODO: Make is_open method so don't have to call this
        if random:
            if safe_mode:
                self.move_to_angles(phi=0.05, along_hemi=False)
                sleep(1)
            fraction = self.move_to_random() 
        else:
            ret = self.move_to_angles(theta=self.reset_angles[0],
                phi=self.reset_angles[1], along_hemi=along_hemi)
            fraction = ret['fraction']
        
        if fraction == 0.0:
            print("Reset failure!")
            print("Fraction: " + str(fraction))
            if kill_on_fail:
                raise RuntimeError('Could not plan reset trajectory')
        
        if verbose:
            print("Done.")
            sys.stdout.flush()

        return self.get_state()
    
    def take_action(self, angle_inc):
        """ Take action with arm. Returns status info dictionary. """
        if type(angle_inc) == np.ndarray:
            angle_inc = angle_inc.reshape((2,)).tolist()
        
        [cur_theta, cur_phi] = self.get_cur_angles() 
        theta = cur_theta + angle_inc[0]
        phi = cur_phi + angle_inc[1]

        self._publish_step(angle_inc) 
        info = self.move_to_angles(theta=theta, phi=phi, along_hemi=False)
        self.step_count += 1
        self._publish_step_count()

        return info 

    def step(self, angle_inc, verbose=True):
        """ Take action with arm and get updated observations.
            Returns observation, reward, done, and info as specified on
            http://gym.openai.com/docs/
        """
        start = time.time()
        #self.check_joint_error()    
        
        info = self.take_action(angle_inc) 
        [obs, pos] = self.get_state()
        reward = self.get_reward(action=angle_inc, in_bounds=info['valid'])

        done = not info['valid'] or self.episode_finished()
        info['elapsed'] = time.time() - start 
        
        if verbose:
            print('| Step: {:d}/{:d} | Episode: {:d} | Reward: {:2f}'.format(
                self.step_count, self.max_episode_steps, self.episode_count,
                reward))
            print('Info: ' + str(info))
            sys.stdout.flush()

        self.last_return = ([obs, pos], reward, done, info)
        return self.last_return