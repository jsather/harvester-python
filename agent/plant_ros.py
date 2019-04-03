""" plant_ros.py contains interface to talk to simulated strawberry plant in
    Gazebo/ROS.

    Author: Jonathon Sather
    Last updated: 9/10/2018

    Note: Having difficulties using subprocess.Popen([arg1, arg2, etc...]), 
        so sending direct command 'arg1 arg2 arg3' with shell=True. 
        Not ideal, but it works for now.
"""
from __future__ import print_function
import os
import sys
import subprocess
import json
import numpy as np
from time import sleep

import rospy 
from gazebo_msgs.srv import DeleteModel

import config as agent_cfg


class PlantROS:
    """ Class for interfacting with strawberry plant through ROS/Gazebo.
        Contains method to spawn new random plant in environment.
    """

    def __init__(self, name='plant1', spawned=False, new_node=False):
        """ Initialize plant object."""
        if new_node:
            rospy.init_node('plant_node')

        self.spawned = spawned
        self.name = name
        self.model_dir = agent_cfg.plant_model_dir 
        self.ripe_file = os.path.join(self.model_dir, 'berry_data.txt')
        self.ripe_data = {'poses': [], 'radii': [], 'ripe': []}

        # Bed params # TODO: add to agent_config
        self.num_rows = 4
        self.bed_corner = [1.6, -0.2263]
        self.bed_dx = -3.0
        self.bed_dy = 1.6
    
    def is_spawned(self):
        """ Returns true if plant is spawned. False otherwise. """
        return self.spawned 
    
    def new(self, x=0.0, y=0.0, remove=True, name=None, verbose=True,
        write=True, plant_model_prefix='model', sdf=None):
        """ Spawns a new plant in Gazebo environment. """
        if remove:
            self.remove()

        if name is None:
            name = self.name

        if verbose:
            print("Sending message to spawn plant " + str(name) +
                " at (x, y) = (" + str(x) + ", " + str(y) + ")... ", end="")
            sys.stdout.flush()
        
        if not sdf: 
            # Generate random plant
            # cmd = ['erb', 'model.rsdf', '>', 'model.sdf'] # Not working - gives no such file or directory @ rb_sysopen error
            rsdf = plant_model_prefix + '.rsdf'
            sdf = plant_model_prefix + '.sdf'

            cmd = 'erb ' + rsdf + ' > ' + sdf
            subprocess.Popen(cmd, shell=True, cwd=self.model_dir)
        
        # Spawn plant 
        cmd = 'rosrun gazebo_ros spawn_model -file ' + sdf + ' -sdf -model ' + \
            name + ' -x ' + str(x) + ' -y ' + str(y) 
        subprocess.Popen(cmd, shell=True, cwd=self.model_dir)
        self.spawned = True

        if verbose:
            print("Done.")

        if write:
            with open(self.ripe_file, 'r') as rf:
                lines = rf.readlines()
                for line in lines:
                    ripe_data = json.loads(line)
                    self.ripe_data['poses'].append(ripe_data[0])
                    self.ripe_data['radii'].append(ripe_data[1])
                    self.ripe_data['ripe'].append(ripe_data[2])

    def new_full_bed(self, verbose=True, write=True):
        """ Spawns full bed of plants in Gazebo environment. Returns names of
            spawned plants.
        """
        spacing =  (2/np.sqrt(2)) * abs(self.bed_dy)/(self.num_rows + 1)
        x_count = np.floor(abs(self.bed_dx / spacing))
        x_odd = np.linspace(self.bed_corner[0] + np.sign(self.bed_dx)*spacing/2,
            self.bed_corner[0] + np.sign(self.bed_dx)*(x_count + 0.5)*spacing,
            num=x_count,
            endpoint=False)
        end_dist = abs((self.bed_corner[0] + self.bed_dx) - x_odd[-1])
        if end_dist > spacing:
            x_even = x_odd[:] + np.sign(self.bed_dx)*spacing/2
        else:
            x_even = x_odd[:-1] + np.sign(self.bed_dx)*spacing/2

        y_vals = np.linspace(
            self.bed_corner[1] + np.sign(self.bed_dy)*(np.sqrt(2)/2)*spacing/2,
            self.bed_corner[1] + np.sign(self.bed_dy)*(np.sqrt(2)/2)*5*spacing,
            num=self.num_rows,
            endpoint=False) # shift left for kicks

        names = []
        num_spawned = 0
        for idx, y in enumerate(y_vals):
            if idx % 2 == 0:
                for x in x_odd:
                    name = 'plant' + str(num_spawned+1)
                    self.new(x=x, y=y, remove=False, name=name, verbose=verbose,
                        write=write)
                    names.append(name)
                    num_spawned += 1
                    sleep(15)
            else:
                for x in x_even:
                    name = 'plant' + str(num_spawned+1)
                    self.new(x=x, y=y, remove=False, name=name, verbose=verbose,
                        write=write)
                    names.append(name)
                    num_spawned += 1
                    sleep(15)

        return names

    def remove(self, name=None, verbose=True):
        """ Removes specified plant from simulated environment. """
        if self.spawned:
            if name is None:
                name = self.name

            if verbose:
                print("Sending message to remove plant " + name + "... ",
                    end="")
                sys.stdout.flush()

            # cmd = ['rosservice', 'call', 'gazebo/delete_model', 
            #     '{model_name: ' + name + '}'] # not working...
            rospy.wait_for_service('gazebo/delete_model', timeout=10)

            try:
                delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                delete_model(name) 
            except rospy.ServiceException, e:
                print("Service call failed: %s", e)
                raise RuntimeError('Unable to remove plant from environment.')
            # cmd = 'rosservice call gazebo/delete_model ' + name
            # subprocess.Popen(cmd, shell=True, cwd=self.model_dir)
            self.spawned = False
            self.ripe_data = {'poses': [], 'radii': [], 'ripe': []}

            if verbose:
                print("Done.")

    def get_ripe_data(self):
        """ Returns poses and radii of all ripe strawberries on current plant.
        """
        return self.ripe_data

if __name__ == '__main__':
    import agent_utils 
    sim =  agent_utils.HarvesterSimulation() 
    sim.run(verbose=False, headless=False, world_only=True) 
    plant = PlantROS(new_node=True)
    
    plant.new()
    sleep(20)
    plant.remove()

    print('Spawning bed')
    plant.new_full_bed()

    while True:
        sleep(15)

    # print('Spawning plants every 15 seconds, until KeyboardInterrupt')

    # try:
    #     while True:
    #         plant.new()
    #         sleep(15)
    # except KeyboardInterrupt:
    #     print('KeyboardInterrupt detected. Moving on.')

    # plant.remove()
    # plant.new_full_bed()

    # try:
    #     while True:
    #         sleep(15)
    # except KeyboardInterrupt:
    #     print('KeyboardInterrupt detected. Goodbye.')
    
    