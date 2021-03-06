"""
    fixation_analysis.py contains scripts for analyzing fixation data, as the
    name might imply.
"""

import ast 
import csv
import glob 
import os 
import signal 
import sys 
import time 

from mpl_toolkits.mplot3d import Axes3D
from sensor_msgs.msg import Image
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import psutil 
import rospy

import agent.config as agent_cfg
import agent.plant_ros as plant_ros
import agent.agent_utils as agent_utils 
import agent.agent_ros as agent_ros 

show_plant = True
display_image = False 
#outdir = '/media/jonathon/JON SATHER/Thesis/results/good_fixations_annotated'
#outdir = '/media/jonathon/JON SATHER/Thesis/results/bad_fixations_annotated'
outdir = '/media/jonathon/JON SATHER/Thesis/results/no_fixations_annotated'

class Camera(object):
    """ ROS camera subscriber. """
    def __init__(self, topic='/harvester/camera2/image_raw'):
        self.sub = rospy.Subscriber(topic, Image, self._process_image_data, queue_size=1)
        self.obs = None 
    
    def _process_image_data(self, ros_data):
        flat = np.fromstring(ros_data.data, np.uint8)
        full = np.reshape(flat, (ros_data.height, ros_data.width, -1))
        self.obs = full[...,::-1] # convert to bgr
        # self.obs = cv2.resize(full, (self.obs_shape[0], self.obs_shape[1]))

def create_hemisphere(radius):
    """ Returns x,y,z data for plotting hemisphere of specified radius. """
    phi, theta = np.mgrid[0:0.5*np.pi:100j, 0.0:2.0*np.pi:100j]
    x = radius*np.sin(phi)*np.cos(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(phi)
    return (x, y, z)

def create_hemi_image(radius):
    x_sph, y_sph, z_sph = create_hemisphere(radius=radius)

    fig = plt.figure(figsize=(8,8), dpi=100)
    ax = Axes3D(fig) 
    ax.set_xlim3d(-radius, radius)
    ax.set_ylim3d(-radius, radius)
    ax.set_zlim3d(-0.2*(2*radius), 0.8*(2*radius))

    ax.view_init(azim=225)
    ax.set_axis_off() 
    ax.plot_surface(x_sph, y_sph, z_sph, rstride=1, cstride=1, 
        color='c', alpha=0.3, linewidth=0)

    plt.savefig('hemi.png', transparent=True)
    plt.close(fig)
    time.sleep(1)
    return cv2.imread('hemi.png')

def create_overlay_image(coords, rewards, radius, hemi, obs=None, name='plant'):
    """ Creates image overlay and saves to file. """
    global outdir 

    x_pos = [] 
    y_pos = [] 
    z_pos = [] 

    x_neu = [] 
    y_neu = [] 
    z_neu = [] 

    x_term = [] 
    y_term = [] 
    z_term = [] 

    x = [] 
    y = [] 
    z = [] 

    for idx, coord in enumerate(coords):
        if rewards[idx] == 1.0:
            x_pos.append(coord[0])
            y_pos.append(coord[1])
            z_pos.append(coord[2])
        elif rewards[idx] == -0.1:
            x_neu.append(coord[0])
            y_neu.append(coord[1])
            z_neu.append(coord[2])
        else:
            x_term.append(coord[0])
            y_term.append(coord[1])
            z_term.append(coord[2])

        x.append(coord[0])
        y.append(coord[1])
        z.append(coord[2])
    
    # plot points
    fig = plt.figure(figsize=(8,8), dpi=100)
    ax = Axes3D(fig) 
    ax.set_xlim3d(-radius, radius)
    ax.set_ylim3d(-radius, radius)
    ax.set_zlim3d(-0.2*(2*radius), 0.8*(2*radius))

    ax.scatter(x_pos, y_pos, z_pos, c='b', marker='*')
    ax.scatter(x_neu, y_neu, z_neu, c='y', marker='o')
    ax.scatter(x_term, y_term, z_term, c='r', marker='o')
    ax.view_init(azim=225+90) #225
    ax.set_axis_off() 

    plt.savefig('plot.png', transparent=True)
    plt.close(fig)
    time.sleep(1)
    plot = cv2.imread('plot.png')            
    
    if obs is not None:
        hemi_gray = cv2.cvtColor(hemi, cv2.COLOR_BGR2GRAY)
        _, hemi_mask = cv2.threshold(hemi_gray, 1, 255, 
            cv2.THRESH_BINARY)
        hemi_mask_inv = 255 - hemi_mask 

        hemi_fg = cv2.bitwise_and(hemi, hemi, mask=hemi_mask) 
        obs_fg = cv2.bitwise_and(obs, obs, mask=hemi_mask)
        obs_bg = cv2.bitwise_and(obs, obs, mask=hemi_mask_inv)

        hemi_blended = cv2.addWeighted(hemi_fg, 0.3, obs_fg, 0.7, 0.0)
        obs_hemi = cv2.add(hemi_blended, obs_bg) 

        plot_gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
        _, plot_mask = cv2.threshold(plot_gray, 1, 255, 
            cv2.THRESH_BINARY)
        plot_mask_inv = 255 - plot_mask 
        
        plot_fg = cv2.bitwise_and(plot, plot, mask=plot_mask)
        obs_hemi_bg = cv2.bitwise_and(obs_hemi, obs_hemi, 
            mask=plot_mask_inv) 
        overlay = cv2.add(plot_fg, obs_hemi_bg)

        cv2.imwrite(os.path.join(outdir, name[:-4] + '_plant' + '.png'), 
            overlay)
    else:
        plot_gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
        _, plot_mask = cv2.threshold(plot_gray, 1, 255, cv2.THRESH_BINARY)
        # _, plot_mask = cv2.threshold(plot_gray, 254, 255,
        #     cv2.THRESH_BINARY_INV)
        plot_mask_inv = 255 - plot_mask 

        plot_fg = cv2.bitwise_and(plot, plot, mask=plot_mask)
        hemi_bg = cv2.bitwise_and(hemi, hemi, mask=plot_mask_inv)
        overlay = cv2.add(plot_fg, hemi_bg)

        cv2.imwrite(os.path.join(outdir, name[:-4] + '_hemi' + '.png'), 
            overlay)

def exit_gracefully(sig, frame):
    """ Save configuration before exit. """
    print('Signal: ' + str(sig))
    kill_processes(['roslaunch', 'gzserver', 'gzclient'])
    sys.exit()

def file_from_path(path):
    """ Extracts local filename from full path """
    slashes = [pos for pos, char in enumerate(path) if char == '/']
    return path[slashes[-1]+1:]

def kill_processes(processes, delay=1):
    """ Kills processes in list. """
    for proc in psutil.process_iter():
        if proc.name in processes:
            print('killing process: ' + proc.name)
            proc.kill() 
            time.sleep(delay)

def spherical_to_cartesian(theta, phi, rho):
    """ Converts spherical coordinates into cartesian. """
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    return x, y, z 

def main():
    global show_plant 

    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    fixation_results_dir = '/media/jonathon/JON SATHER/Thesis/results/fixation_test_extended'
    test_dirs = glob.glob(fixation_results_dir + '/*')

    rewards = []
    coords = []
    plant_files = []
    good_list = [65, 366, 309, 18, 42] #[122, 327, 308, 195, 189]  
    bad_list = [196, 178, 149, 95, 74] #[307, 379, 390, 422, 424] 
    no_list = [219, 314, 336, 186, 93] #[3, 14, 227, 255, 269] 

    plant_list = ['model' + str(num) + '.sdf' for num in no_list]

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

    if show_plant:
        sim = agent_utils.HarvesterSimulation()
        sim.cfg['paused'] = False 
        sim.run(world_only=True)
        
        rospy.init_node('camera')
        camera = Camera()
        plant = plant_ros.PlantROS()

    hemi = create_hemi_image(radius=agent_cfg.hemi_radius)
    
    for plant_no in range(len(coords)):
        plant_name = file_from_path(plant_files[plant_no])
        if plant_name in plant_list:
        # if len(coords[plant_no]) > 50:
            # spawn plant and take image
            if show_plant:
                plant.new(sdf=plant_files[plant_no])
                time.sleep(30)
                obs = camera.obs
            else:
                obs = None
 
            create_overlay_image(coords=coords[plant_no], rewards=rewards[plant_no], radius=agent_cfg.hemi_radius, hemi=hemi, obs=obs, name=plant_name)
    
    os.kill(os.getpid(), signal.SIGTERM)

if __name__ == "__main__":
    main()