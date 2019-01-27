"""
    fixation_analysis.py contains scripts for analyzing fixation data, as the
    name might imply.
    
    author: Jonathon Sather
    last updated: 1/24/2019
"""

# fixation analysis

# so then, what remains is
# - get videos of each of the policies,
# - get screenshots when it seems like uncovering berries (with and without info)
# - make visual showing both fixation behavior and not - discard random policy
# - comment that speculate that this behavior is due to fixation and human
#   logic, but that it is unreliable due to not being able to communicate
# - based on both the performance and the appearance it is unlikely that there is
#   any propagation through time that is super beneficial - could just be responding
#   to other clues at these instances

import ast 
import csv
import glob 
import os 
import sys 
import time 

from mpl_toolkits.mplot3d import Axes3D
from sensor_msgs.msg import Image
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import rospy

import agent.config as agent_cfg
import agent.plant_ros as plant_ros
import agent.agent_utils as agent_utils 

import pdb 

class Camera(object):
    """ ROS camera subscriber. """
    def __init__(self, topic='/harvester/camera1/image_raw'):
        self.sub = rospy.Subscriber(topic, Image, self._process_image_data, queue_size=1)
        self.obs = None 
    
    def _process_image_data(self, ros_data):
        flat = np.fromstring(ros_data.data, np.uint8)
        full = np.reshape(flat, (ros_data.height, ros_data.width, -1))
        self.obs = full[...,::-1] # convert to bgr
        # self.obs = cv2.resize(full, (self.obs_shape[0], self.obs_shape[1]))

def spherical_to_cartesian(theta, phi, rho):
    """ Converts spherical coordinates into cartesian. """
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    return x, y, z 

def main():
    fixation_results_dir = '/media/jonathon/JON SATHER/Thesis/results/fixation_test_extended'
    test_dirs = glob.glob(fixation_results_dir + '/*')
    
    # analyze ddpg first - then see if want to make functions
    # extract rewards and positions into numpy arrays
    # rewards = np.empty([len(test_dirs), 100], dtype=float) 
    # positions = np.empty([len(test_dirs), 100, 2], dtype=float)
    
    rewards = []
    coords = []
    plant_files= []

    while test_dirs:
        earliest = min(test_dirs, key=os.path.getctime)

        for fn in glob.glob(os.path.join(earliest, '*')):
            if fn[-3:] == 'sdf':
                spaces = [pos for pos, char in enumerate(fn) if char == ' ']
                for pos in spaces:
                    fn = fn[:pos] + '\ ' + fn[pos+1:]
                plant_files.append(fn)
                break 

        with open(os.path.join(earliest, 'ddpg.csv')) as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                rewards.append(ast.literal_eval(row['all_rewards']))
                hemi_coords = ast.literal_eval(row['all_j'])
                
                cart_coords = []
                for sph in hemi_coords:
                    (x, y, z) = spherical_to_cartesian(sph[0], sph[1], 
                        agent_cfg.hemi_radius)
                    cart_coords.append((x,y,z))
                # convert to xyz
                coords.append(cart_coords)
        test_dirs.remove(earliest)

    # plot coordinates? try it out!
    # TODO: Figure out equivalent camera position
    #       Figure out how to exit script without crashing
    #       Figure out how to plot without noise
    #       Figure out good plots to use and corresponding plants - display plant with plot in title!
    #       Collect plant images from camera angle (autonomously?) and create cool images for report!
    #       Figure out optimal camera angle

    sim = agent_utils.HarvesterSimulation()
    sim.cfg['paused'] = False 
    sim.run(world_only=True)
    
    rospy.init_node('camera')
    camera = Camera()
    plant = plant_ros.PlantROS()

    for plant_no in range(len(coords)):
        pts = len(coords[plant_no])
        if pts > 50:
            # spawn plant and take image
            plant.new(sdf=plant_files[plant_no])
            time.sleep(30)

            x_pos = [] #np.empty([pts])
            y_pos = [] #np.empty([pts])
            z_pos = [] #np.empty([pts])
            # pos_idx = 0

            x_neu = [] #np.empty([pts])
            y_neu = [] #np.empty([pts])
            z_neu = [] #np.empty([pts])
            # neu_idx = 0

            x_term = [] #np.empty([pts])
            y_term = [] #np.empty([pts])
            z_term = [] #np.empty([pts])
            # term_idx = 0

            x = [] #np.empty([pts])
            y = [] #np.empty([pts])
            z = [] #np.empty([pts])

            for idx, coord in enumerate(coords[plant_no]):
                if rewards[plant_no][idx] == 1.0:
                    # x_pos[pos_idx] = coord[0]
                    # y_pos[pos_idx] = coord[1]
                    # z_pos[pos_idx] = coord[2]
                    # pos_idx += 1
                    x_pos.append(coord[0])
                    y_pos.append(coord[1])
                    z_pos.append(coord[2])
                elif rewards[plant_no][idx] == -0.1:
                    # x_neu[neu_idx] = coord[0]
                    # y_neu[neu_idx] = coord[1]
                    # z_neu[neu_idx] = coord[2]
                    # neu_idx += 1
                    x_neu.append(coord[0])
                    y_neu.append(coord[1])
                    z_neu.append(coord[2])
                else:
                    # x_term[term_idx] = coord[0]
                    # y_term[term_idx] = coord[1]
                    # z_term[term_idx] = coord[2]
                    # term_idx += 1
                    x_term.append(coord[0])
                    y_term.append(coord[1])
                    z_term.append(coord[2])

                # x[idx] = coord[0]
                # y[idx] = coord[1]
                # z[idx] = coord[2]
                x.append(coord[0])
                y.append(coord[1])
                z.append(coord[2])
            
            fig = plt.figure(figsize=(8,8), dpi=100)
            # plt.axes('off')
            #ax.plot(x, y, zs=z, c='y')#, c='b', marker='o')
            # ax2.axes('off')
            ax = Axes3D(fig) #fig.add_subplot(111, projection='3d')
            ax.set_xlim3d(-agent_cfg.hemi_radius, agent_cfg.hemi_radius)
            ax.set_ylim3d(-agent_cfg.hemi_radius, agent_cfg.hemi_radius)
            ax.set_zlim3d(0, agent_cfg.hemi_radius)
            ax.scatter(x_pos, y_pos, z_pos, c='b', marker='*')
            ax.scatter(x_neu, y_neu, z_neu, c='y', marker='o')
            ax.scatter(x_term, y_term, z_term, c='r', marker='o')
            ax.view_init(azim=65)
            ax.set_axis_off() # comment when aligned!
            plt.savefig('plot.png', transparent=True)
            time.sleep(1)
            plot = cv2.imread('plot.png')
            
            plot_gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
            _, plot_mask = cv2.threshold(plot_gray, 254, 255, 
                cv2.THRESH_BINARY_INV)
            foreground = cv2.bitwise_and(plot, plot, mask=plot_mask)

            obs = camera.obs 
            obs_mask = 255 - plot_mask 
            background = cv2.bitwise_and(obs, obs, mask=obs_mask) 
            
            overlay = cv2.add(foreground, background)

            # dst = cv2.addWeighted(camera.obs, 1, plot, 0.3, 0)
            cv2.imshow('overlay', overlay)
            cv2.waitKey(5000)
            # cv2.destroyAllWindows()

            # ax2 = fig.add_subplot(111,frame_on=False)
            # ax2.imshow(camera.obs)
            # # ax.axes('off')
            # plt.show()

if __name__ == "__main__":
    main()