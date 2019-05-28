"""
    position_analysis.py contains scripts for analyzing position data, as the
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
outdir = '/media/jonathon/JON SATHER/Thesis/results/fixation_obs'

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

def arm_over_existing(obs, hemi, name='plant'):
    """ Overlays image with arm over previously annotated hemi. """      

    hemi_gray = cv2.cvtColor(hemi, cv2.COLOR_BGR2GRAY)
    _, hemi_mask = cv2.threshold(hemi_gray, 1, 255, 
        cv2.THRESH_BINARY)
    hemi_mask_inv = 255 - hemi_mask 

    obs_hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
    _, arm_mask = cv2.threshold(obs_hsv[:,:,1], 3, 255, cv2.THRESH_BINARY_INV)
    arm_mask_inv = 255 - arm_mask
    arm_fg = cv2.bitwise_and(obs, obs, mask=arm_mask)
    arm_fg_hemi = cv2.bitwise_and(arm_fg, arm_fg, mask=hemi_mask)
    arm_fg_no_hemi = cv2.bitwise_and(arm_fg, arm_fg, mask=hemi_mask_inv)

    existing = cv2.imread(os.path.join(outdir, name[:-4] + '_plant' + '.png'))
    existing_fg = cv2.bitwise_and(existing, existing, mask=arm_mask)
    existing_bg = cv2.bitwise_and(existing, existing, mask=arm_mask_inv)
    existing_fg_hemi = cv2.bitwise_and(existing_fg, existing_fg, mask=hemi_mask)

    arm_blend_hemi = cv2.addWeighted(arm_fg_hemi, 0.6, existing_fg_hemi, 0.4, 0.0)
    arm_blend = cv2.add(arm_blend_hemi, arm_fg_no_hemi) 

    overlay = cv2.add(existing_bg, arm_blend)

    cv2.imwrite(os.path.join(outdir, name[:-4] + '_plant_arm_blend_hemi' + '.png'),
        overlay)

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

def create_overlay_image(coords, rewards, obs, radius, hemi, name='plant'):
    """ Creates image overlay and saves to file. """
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

    hemi_gray = cv2.cvtColor(hemi, cv2.COLOR_BGR2GRAY)
    _, hemi_mask = cv2.threshold(hemi_gray, 1, 255, 
        cv2.THRESH_BINARY)
    hemi_mask_inv = 255 - hemi_mask 

    hemi_fg = cv2.bitwise_and(hemi, hemi, mask=hemi_mask) 
    
    obs_hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
    _, arm_mask = cv2.threshold(obs_hsv[:,:,1], 3, 255, cv2.THRESH_BINARY_INV)
    arm_mask_inv = 255 - arm_mask
    arm_fg = cv2.bitwise_and(obs, obs, mask=arm_mask)

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

    overlay_orig = cv2.add(plot_fg, obs_hemi_bg)
    
    overlay_fg = cv2.bitwise_and(overlay_orig, overlay_orig,
        mask=arm_mask)
    overlay_bg = cv2.bitwise_and(overlay_orig, overlay_orig,
        mask=arm_mask_inv)
    
    overlay = cv2.add(overlay_bg, arm_fg)

    cv2.imwrite(os.path.join(outdir, name[:-4] + '_plant_arm_blend' + '.png'),
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
    # plant_list = ['model19.sdf', 'model84.sdf', 'model424.sdf',
    #     'model161.sdf', 'model309.sdf','model347.sdf', 'model363.sdf',
    #     'model49.sdf', 'model51.sdf', 'model107.sdf', 'model355.sdf', 
    #     'model423.sdf']
    # plant_list = ['model424.sdf',
    #     'model51.sdf', 
    #     'model423.sdf']
    plant_list = ['model122.sdf', 'model308.sdf', 'model74.sdf', 'model149.sdf']
    pos_list = [(-0.6, 1.15), (-0.65, 1.4), (-.12, 1.11), (-1.6, 0.4)]

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


    agent = agent_ros.HemiAgentROS(detector=False)
    agent.move_to_angles(theta=-1.0, phi=0.65)
    camera = Camera()

    # create and save hemi backdrop image
    hemi = create_hemi_image(radius=agent_cfg.hemi_radius)
    
    for plant_no in range(len(coords)):
        plant_name = file_from_path(plant_files[plant_no])
        if plant_name in plant_list:
            # spawn plant and take image
            agent.plant.new(sdf=plant_files[plant_no])
            time.sleep(25)

            (theta, phi) = pos_list[plant_list.index(plant_name)]
            agent.move_to_angles(theta=theta, phi=phi)
            time.sleep(5)
            

            cv2.imwrite(
                os.path.join(outdir, plant_name[:-4] + '_obs' + '.png'), 
                agent.obs[...,::-1])
            arm_over_existing(obs=camera.obs, hemi=hemi, name=plant_name)
            # create_overlay_image(coords=coords[plant_no],
            #     rewards=rewards[plant_no], obs=camera.obs,
            #     radius=agent_cfg.hemi_radius, hemi=hemi, name=plant_name)
    
    exit_gracefully(sig='program  end', frame=None)

if __name__ == "__main__":
    main()