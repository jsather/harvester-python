""" agent_utils.py contains utilities for the agent_ros and plant_ros modules.

    Author: Jonathon Sather
    Last updated: 10/01/2018
"""

import os
import psutil
import shlex
import signal 
import subprocess
import sys 
import time 

import numpy as np
import rospy
import rospkg
from gazebo_msgs.srv import SetModelConfiguration 
from std_srvs.srv import Empty
from controller_manager_msgs.srv import SwitchController 
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.srv import GetPositionIKRequest
from moveit_msgs.srv import GetPositionIKResponse

import config as agent_cfg

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """ Kills child processes of given pid. Code copied from:
        https://answers.ros.org/question/215600/how-can-i-run-roscore-from-python/
    """
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        print("Parent process does not exist.")
        return

    try:
        children = parent.get_children(recursive=True) # Python 2.7.6
    except AttributeError:
        children = parent.children(recursive=True) # Python 2.7.13

    for process in children:
        print("Killing child process: " + str(process))
        process.send_signal(sig)

def add_pid(pid):
    """ Adds pid to stored list of running pids for agent. """
    with open('/tmp/agent.pid', 'a') as f:
        f.write(str(pid) + '\n')

def roslaunch(package, launchfile, args):
    """ Helper function to run launch files. Returns process object. """
    rospack = rospkg.RosPack()
    launch_path = os.path.join(rospack.get_path(package), 'launch', launchfile)
    # cmd = 'nohup roslaunch ' + launch_path
    cmd = 'roslaunch ' + launch_path 

    for key, value in args.iteritems():
        cmd += ' ' + key + ':=' + str(value)

    return subprocess.Popen(shlex.split(cmd), preexec_fn=os.setpgrp)  #Do I want this?

class HarvesterSimulation(object):
    """ Class for launching and stopping harvester ros simulation. Inspired by:
        https://answers.ros.org/question/215600/how-can-i-run-roscore-from-python/
    """

    __initialized = False
    def __init__(self):
        """ Raise exception if multiple Simulation classes created. """
        if HarvesterSimulation.__initialized:
            raise Exception(
                "Roscore instance already created, only allowed one instance.")
        HarvesterSimulation.__initialized = True
        
        self.harvester_project = agent_cfg.harvester_ros_path
        self.cfg = agent_cfg.harvester_default_cfg 
        self.joints = agent_cfg.joint_names # TODO: Add to cfg
        self.joint_angles_init = agent_cfg.joint_angles_init.tolist() # TODO: Add to cfg
        self.running_processes = []

        self.launch_world_delay = agent_cfg.launch_world_delay
        self.load_robot_delay = agent_cfg.load_robot_delay
        self.spawn_controllers_delay = agent_cfg.spawn_controllers_delay 
        self.spawn_robot_delay = agent_cfg.spawn_robot_delay 
        self.move_home_delay = agent_cfg.move_home_delay 
        self.launch_moveit_delay = agent_cfg.launch_moveit_delay 

    def run(self, headless=False, verbose=True, world_only=False):
        """ Start the ROS/Gazebo harvester simulation. """
        if headless:
            self.cfg['gui'] = False 
        
        if verbose:
            print("Starting harvester simulation with configuration: " + 
                str(self.cfg))
        
        self.launch_world(verbose=verbose)
        if world_only:
            return

        # self.load_robot(verbose=verbose)    
        self.spawn_robot(verbose=verbose) # moved robot load into spawn file
        self.spawn_controllers(verbose=verbose)
        self.set_robot_config(verbose=verbose)
        self.unpause_physics(verbose=verbose) 
        # self.start_controllers(verbose=verbose) 
        self.move_home(verbose=verbose)
        self.launch_moveit(verbose=verbose)

    def launch_world(self, verbose=True):
        """ Launches harvester world. """
        if verbose: 
            print("Launching world")
        
        process = roslaunch('harvester_gazebo', 'harvester_world.launch', self.cfg) 
        self.running_processes.append(process) 
        rospy.sleep(self.launch_world_delay)
    
    def load_robot(self, verbose=True):
        """ Loads robot into parameter server. """
        if verbose:
            print("Loading robot")
        
        load_args = {'harvester_robotName': self.cfg['harvester_robotName']}
        process = roslaunch('harvester_gazebo', 'load_robot.launch', load_args) 
        self.running_processes.append(process) 
        rospy.sleep(self.load_robot_delay)
    
    def spawn_controllers(self, verbose=True):
        """ Spawns (stopped) controllers. """
        if verbose: 
            print("Spawning controllers")

        control_args = {'harvester_robotName': self.cfg['harvester_robotName'],
            'kinova_robotName': self.cfg['kinova_robotName'],
            'stopped': False}
        process = roslaunch('harvester_control', 'harvester_control.launch',
            control_args) 
        self.running_processes.append(process)
        rospy.sleep(self.spawn_controllers_delay)
    
    def spawn_robot(self, verbose=True):
        """ Spawns robot. """
        if verbose:
            print("Spawning robot")
        
        spawn_args = {'harvester_robotName': self.cfg['harvester_robotName'],
            'kinova_robotName': self.cfg['kinova_robotName']}
        process = roslaunch('harvester_gazebo', 'spawn_robot.launch', 
            spawn_args)
        self.running_processes.append(process)
        rospy.sleep(self.spawn_robot_delay)
    
    def move_home(self, verbose=True):
        """ Runs script to home robot. """
        if verbose:
            print("Moving home")
        
        home_args = {'kinova_robotName': self.cfg['kinova_robotName']}
        process = roslaunch('harvester_control', 'move_home.launch', home_args) 
        self.running_processes.append(process) 
        rospy.sleep(self.move_home_delay)
    
    def launch_moveit(self, verbose=True):
        """ Launches moveit. """
        if verbose:
            print("Launching Moveit")
        
        moveit_args = {}
        process = roslaunch('harvester_moveit', 'harvester_moveit.launch',
            moveit_args)
        self.running_processes.append(process) 
        rospy.sleep(self.launch_moveit_delay)

    def set_robot_config(self, verbose=True):
        """ Sets model configuration. """
        if verbose:
            print("Setting robot configuration.")

        config_srv = '/gazebo/set_model_configuration'
        rospy.wait_for_service(config_srv)
        set_config = rospy.ServiceProxy(config_srv, SetModelConfiguration)
        ret = set_config(self.cfg['kinova_robotName'], '', self.joints, 
            self.joint_angles_init) #[0,2.9,1.3,4.2,1.4,0.0]) 
        print(ret) 

    def start_controllers(self, verbose=True):
        """ Starts controllers for specified model. Assumes that the controllers
            are loaded but stopped. ['python', '-m', 'agent.start_agent']
        """
        if verbose:
            print("Starting controllers.")
        
        controllers = [
            'joint_state_controller', 
            'effort_joint_trajectory_controller', 
            'effort_finger_trajectory_controller']
        
        switch_srv = os.path.join('/' + self.cfg['kinova_robotName'],
            'controller_manager', 
            'switch_controller')
        rospy.wait_for_service(switch_srv)
        switch_controller = rospy.ServiceProxy(switch_srv, SwitchController)
        ret = switch_controller(controllers, [], 2) 
        print(ret) # TODO: Get rid of this and add while loop!

    def unpause_physics(self, verbose=True):
        """ Unpauses physics in Gazebo simulation. """
        if verbose:
            print("Unpausing physics")

        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause()

    def kill(self):
        """ Kill the ROS/Gazebo simulation. """
        if not self.running_processes:
            print('Simulation not running, cannot kill process.')
            return
        
        print('Killing ' + str(len(self.running_processes)) + ' processes')
        for process in self.running_processes: 
            kill_child_processes(process.pid)
            process.terminate() 
            process.wait() 
        
        self.running_processes = []
        HarvesterSimulation.__initialized = False 

class Feed(object):
    """ Class for launching and stopping annotated camera feed. Inspired by:
        https://answers.ros.org/question/215600/how-can-i-run-roscore-from-python/
    """

    __initialized = False
    def __init__(self):
        """ Raise exception if multiple Feed classes created. """
        if Feed.__initialized:
            raise Exception(
                "Feed already created, only allowed one instance.")
        Feed.__initialized = True

        self.delay = agent_cfg.init_feed_delay 

    def run(self):
        """ Start the feed. """
        try:
            self.feed_process = subprocess.Popen(
                ['python', '-m', 'image.show_feed'])
            self.feed_pid = self.feed_process.pid
            time.sleep(self.delay)
        except OSError as e:
            sys.stderr.write('Feed startup script could not be run.')
            raise e
        self.running = True

    def kill(self):
        """ Kill the feed. """
        if not self.running:
            print("Feed not running, cannot kill process.")
            return

        print("Killing children of feed pid: " + str(self.feed_pid))
        kill_child_processes(self.feed_pid)

        print("Killing parent process: " + str(psutil.Process(self.feed_pid)))
        self.feed_process.terminate()
        self.feed_process.wait() # Apparently prevents "zombie process"
        Feed.__initialized = False

class GetIK(object):
    """ Class for creating IK calls in Moveit.
        Created by Sammy Pfeiffer.
        Source: https://github.com/uts-magic-lab/moveit_python_tools/blob/master/src/moveit_python_tools/get_ik.py
    """
    def __init__(self, group, ik_timeout=1.0, ik_attempts=0,
        avoid_collisions=False, verbose=False):
        """
        A class to do IK calls thru the MoveIt!'s /compute_ik service.
        :param str group: MoveIt! group name
        :param float ik_timeout: default timeout for IK
        :param int ik_attempts: default number of attempts
        :param bool avoid_collisions: if to ask for IKs that take
        into account collisions
        """
        if verbose:
            rospy.loginfo("Initalizing GetIK...")

        self.group_name = group
        self.ik_timeout = ik_timeout
        self.ik_attempts = ik_attempts
        self.avoid_collisions = avoid_collisions

        if verbose:
            rospy.loginfo("Computing IKs for group: " + self.group_name)
            rospy.loginfo("With IK timeout: " + str(self.ik_timeout))
            rospy.loginfo("And IK attempts: " + str(self.ik_attempts))
            rospy.loginfo("Setting avoid collisions to: " +
                          str(self.avoid_collisions))
        self.ik_srv = rospy.ServiceProxy('/compute_ik',
                                         GetPositionIK)
        if verbose:
            rospy.loginfo("Waiting for /compute_ik service...")
        self.ik_srv.wait_for_service()

        if verbose:
            rospy.loginfo("Connected!")

    def get_ik(self, pose_stamped,
               seed=None,
               group=None,
               ik_timeout=None,
               ik_attempts=None,
               avoid_collisions=None):
        """
        Do an IK call to pose_stamped pose.
        :param geometry_msgs/PoseStamped pose_stamped: The 3D pose
            (with header.frame_id)
            to which compute the IK.
        :param str group: The MoveIt! group.
        :param float ik_timeout: The timeout for the IK call.
        :param int ik_attempts: The maximum # of attemps for the IK.
        :param bool avoid_collisions: If to compute collision aware IK.
        """
        if group is None:
            group = self.group_name
        if ik_timeout is None:
            ik_timeout = self.ik_timeout
        if ik_attempts is None:
            ik_attempts = self.ik_attempts
        if avoid_collisions is None:
            avoid_collisions = self.avoid_collisions

        req = GetPositionIKRequest()
        if seed is not None:
            req.ik_request.robot_state = seed
        req.ik_request.group_name = group
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.timeout = rospy.Duration(ik_timeout)
        req.ik_request.attempts = ik_attempts
        req.ik_request.avoid_collisions = avoid_collisions

        try:
            resp = self.ik_srv.call(req)
            return resp
        except rospy.ServiceException as e:
            rospy.logerr("Service exception: " + str(e))
            resp = GetPositionIKResponse()
            resp.error_code = 99999  # Failure
        return resp
