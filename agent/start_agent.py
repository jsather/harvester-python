""" Script for starting agent_ros node.
    Author: Jonathon Sather
    Last updated: 9/28/18
"""

import argparse
import ast
import collections
import time
import warnings

import cv2
import numpy as np
import rospy
import std_msgs.msg

import agent.agent_ros as agent
import agent.config as agent_cfg

import pdb

class AgentNode(object):
    """ Class for creating and running HemiAgentROS node. """

    def __init__(self, headless=False, feed=False, detector=True):
        """ Initialize agent and publishers. """
        # NOTE: ROS node initialized by HemiAgentROS
        self.agent = agent.HemiAgentROS(headless=headless, feed=feed,
            detector=detector)
        self.command = collections.deque(maxlen=agent_cfg.command_buffer_size)

        namespace = '/' + self.agent.model_name + '/'
        self.state_pub = rospy.Publisher(
            namespace + 'state', std_msgs.msg.String, queue_size=1)
        self.step_return_pub = rospy.Publisher(
            namespace + 'return', std_msgs.msg.String, queue_size=1)
        self.command_sub = rospy.Subscriber(
            namespace + 'command', std_msgs.msg.String, self._store_command,
            queue_size=1)

    def _store_command(self, ros_data):
        """ Callback function for command topic. Receives string specifying
            agent command, and stores information in memberdata.
        """
        self.command.append(eval(ros_data.data))

    def handle_command(self):
        """ Handles oldest command stored in queue. """
        cmd = self.command.popleft()

        if cmd['method'] == 'step':
            [o, p], r, t, i = self.agent.step(*cmd['args'])
            o_str = o.tostring()
            msg = std_msgs.msg.String(str([[o_str, p], r, t, i]))
            self.step_return_pub.publish(msg)
        elif cmd['method'] == 'reset':
            o, p = self.agent.reset(*cmd['args'])
            o_str = o.tostring()
            msg = std_msgs.msg.String(str([o_str, p]))
            self.state_pub.publish(msg)
        else:
            warnings.warn('Invalid method recieved on command topic: ' +
                str(cmd), Warning)

    def run(self):
        """ Continually recieve and execute commands. """
        while not rospy.is_shutdown():
            if self.command:
                self.handle_command()
            if not self.command:
                time.sleep(0.1)

def main():
    """ Run agent node. """
    parser = argparse.ArgumentParser(
        description='provde args for agent node')
    parser.add_argument('--headless',
        help='simulation and camera feed headless',
        default=False,
        action='store_true')
    args = parser.parse_args()

    agent_node = AgentNode(headless=args.headless, feed=(not args.headless))
    agent_node.run()

if __name__ == '__main__':
    main()
else:
    main() #lol
