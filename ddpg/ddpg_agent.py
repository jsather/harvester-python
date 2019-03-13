""" DDPG agent node. 
    Author: Jonathon Sather
    Last updated: 9/28/18
"""
import psutil 
import signal 
import subprocess 
import sys 
import time 

import numpy as np 
import rospy 
import std_msgs.msg 

import agent.config as agent_cfg 
import config as ddpg_cfg 

import pdb 

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

class DDPGAgent(object):
    """ Class for creating and running HemiAgentROS node. """
    
    def __init__(self, headless=False, feed=False, detector=True):
        """ Initialize publishers and subscribers. """
        self.state = None 
        self.ret = None

        rospy.init_node('ddpg_agent_node')

        namespace= '/' + agent_cfg.model_name + '/'
        self.command_pub = rospy.Publisher( 
            namespace + 'command', std_msgs.msg.String, queue_size=1)
        self.state_sub = rospy.Subscriber(
            namespace + 'state', std_msgs.msg.String, self._get_state, 
            queue_size=1)
        self.step_return_sub = rospy.Subscriber( 
            namespace + 'return', std_msgs.msg.String, self._get_return,
            queue_size=1)

        self.start_agent_ros(headless=headless)

    def _get_state(self, ros_data):
        """ Callback function for state topic. Receives string specifying
            state, and stores information in memberdata. 
        """
        [o_str, p] = eval(ros_data.data)
        o_flat = np.fromstring(o_str, dtype=np.uint8)
        o = np.reshape(o_flat, agent_cfg.obs_shape)
        self.state= [o, p]
    
    def _get_return(self, ros_data):
        """ Callback function for return topic. Receives string specifying 
            return, and stores information in memberdata. 
        """
        [[o_str, p], r, t, i] = eval(ros_data.data)
        o_flat = np.fromstring(o_str, dtype=np.uint8)
        o = np.reshape(o_flat, agent_cfg.obs_shape)
        self.ret = [[o, p], r, t, i]
    
    def kill_agent_ros(self):
        """ Kills AgentROS node. """
        try: 
            kill_child_processes(self.agent_ros_process.pid) 
        except: 
            pass 
            
    def start_agent_ros(self, headless=True):
        """ Starts AgentROS node. """
        try: 
            cmd = ['python', '-m', 'agent.start_agent']
            if headless: 
                cmd.append('--headless')
            self.agent_ros_process = subprocess.Popen(cmd)
        except OSError as e: 
            sys.stderr.write('Agent node could not be run.')
            raise e
            
        time.sleep(agent_cfg.total_delay)
    
    def step(self, action):
        """ Sends step command to agent node """
        if type(action) == np.ndarray:
            action = action.reshape((2,)).tolist()

        msg = str({'method': 'step', 'args': (action,)})
        self.command_pub.publish(msg)
        
    def reset(self):
        """ Sends reset command to agent node. """
        msg = str({'method': 'reset', 'args': ()})
        self.command_pub.publish(msg)
    
    def get_state(self):
        """ Returns stored state and clears memberdata. """
        if self.state: 
            state = self.state 
            self.state = None 
        else: 
            state = None 
        return state 
    
    def get_return(self):
        """ Returns stored return and clears memberdata. """
        if self.ret:
            ret = self.ret 
            self.ret = None 
        else: 
            ret = None 
        return ret 