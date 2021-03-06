""" policy.py contains methods to evaluate different policies in
    harvester environment, including agent policy learned by DDPG and a hard
    coded "expert" policy.

    Author: Jonathon Sather
    Last updated: 10/21/2018

    TODO:
   Done?
    Comparisons: 
      1. Try with 3 custom agents. One that moves down and does proportional,
         one that moves down and freezes, one that random and freezes. actually 4-
         one that moves randomly but tries to stay in bounds... hope this isn't
         the best.
      2. Next try with lower density of strawberries
"""
import argparse 
import copy 
import csv
import datetime 
import glob 
import os
import pickle 
import random 
import signal 
import sys
import time

import cv2 
import numpy as np
import psutil
import tensorflow as tf

import agent.agent_ros as agent_ros
import agent.config as agent_cfg
import config as ddpg_cfg 
import networks 
import noise 

import pdb

#------------------------------ Helper functions ------------------------------#
def get_latest_weights(weights_dir):
    """ Gets latest weights file and var pickle from specified directory. """
    dir_contents = glob.glob(weights_dir + '/*')
    while dir_contents:
        latest = max(dir_contents, key=os.path.getctime)
        weights_file_list = glob.glob(latest + '/ddpg*.meta') 

        if weights_file_list: 
            return max(weights_file_list, key=os.path.getctime)[:-5]
        dir_contents.remove(latest)
    return ''

def kill_named_processes(name, keep=[], sig=signal.SIGKILL):
    """ Kills all processes with given name, except those with pids in keep
        list. 
    """
    try: # Python 2.7.6
        running = psutil.get_pid_list() 
    except AttributeError: # Python 2.7.13
        running = psutil.pids() 

    for p in running: 
        try:
            process = psutil.Process(p)
            if process.name() == name and p not in keep: 
                process.send_signal(sig)
        except: # Will get exception if process already terminated
            pass 

#-------------------------------- Classes -------------------------------------#
class Policy(object):
    """ General policy object. """

    def __init__(self):
        """ Initialize the policy object. """
        pass

    def get_action(self, o, j):
        raise NotImplementedError('Must be implemented in child class.')

class CustomPolicy1(Policy):
    """ This custom policy combines 'downward' heuristic with information from
        strawberry detector.
    """

    def __init__(self, action_bound, detector, lut_info, local_p=1.0, 
        thresh=0.5):
        """ Initialize custom policy."""
        super(Policy, self).__init__() 
        self.action_bound = action_bound 
        self.detector = detector
        self.lut_thetas = lut_info['thetas'] 
        self.lut_phis = lut_info['phis'] 
        self.lut_mask = lut_info['mask']  
        self.p = local_p  
        self.thresh = thresh 
        self.name = 'down + proportional + in-bounds'

        self.im_ctr = (agent_cfg.obs_shape[0]/2.0, agent_cfg.obs_shape[1]/2.0)
    
    def best_bb(self, bbs):
        """ Returns predicted bounding box for ripe strawberry with highest
            confidence value. 
        """
        best = None 
        max_confidence = 0.0 

        for bb in bbs:
            if bb[0] and (bb[1] > max_confidence):
                max_confidence = bb[1] 
                best = copy.copy(bb)
        
        return best 
    
    def angle_check(self, current_angles, desired_angles):
        """ Determines validity of each angle by checking lut. Returns 
            (bool theta_check, bool phi_check). 
        """
        theta = desired_angles[0] % (2*np.pi) 
        phi = desired_angles[1]
        theta_idx = (np.abs(self.lut_thetas - theta)).argmin() 
        phi_idx = (np.abs(self.lut_phis - phi)).argmin() 

        cur_theta = current_angles[0] % (2*np.pi)
        cur_phi = current_angles[1] 
        cur_theta_idx = (np.abs(self.lut_thetas - cur_theta)).argmin() 
        cur_phi_idx = (np.abs(self.lut_phis - cur_phi)).argmin() 

        if phi < 0.0 or phi > np.pi/2:   
            theta_valid = self.lut_mask[theta_idx, cur_phi_idx]
            phi_valid = False 
        elif self.lut_mask[theta_idx, phi_idx]:
            theta_valid = True 
            phi_valid = True
        elif self.lut_mask[theta_idx, cur_phi_idx]:
            theta_valid = True 
            phi_valid = False 
        elif self.lut_mask[cur_theta_idx, phi_idx]: 
            theta_valid = False 
            phi_valid = True 
        else: 
            theta_valid = False 
            phi_valid = False 

        return (theta_valid, phi_valid)  

    def get_action(self, o, j):
        """ Compute action given observed camera image and position on 
            hemisphere. 
        """
        bgr = cv2.cvtColor(o,  cv2.COLOR_RGB2BGR)
        bbs = self.detector.detect(bgr)
        best_bb = self.best_bb(bbs)
        
        if best_bb is not None: 
            x_frac = (best_bb[2][1] - self.im_ctr[0]) / self.im_ctr[0]
            y_frac = (best_bb[2][0] - self.im_ctr[1]) / self.im_ctr[1] 
            
            if abs(x_frac) > self.thresh:
                phi_inc = np.sign(x_frac) * self.action_bound[1] 
            else: 
                phi_inc = self.p * x_frac * self.action_bound[1]
            
            if abs(y_frac) > self.thresh: 
                theta_inc = np.sign(y_frac) * self.action_bound[0] 
            else: 
                theta_inc = self.p * y_frac * self.action_bound[1] 
        else: # random theta + down
            theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
            #phi_inc = self.action_bound[1] * (2*random.random() - 1.0)
            phi_inc = self.action_bound[1]

        desired_angles = (j[0] + theta_inc, j[1] + phi_inc)
        current_angles = (j[0], j[1]) 
        theta_valid, phi_valid = \
            self.angle_check(current_angles, desired_angles) 
        
        if not theta_valid: 
            #theta_inc = 0.0 
            theta_inc = -theta_inc 

        if not phi_valid: 
            #phi_inc = 0.0 
            phi_inc = -phi_inc 

        return (theta_inc, phi_inc) 

class CustomPolicy2(Policy):
    """ This custom policy combines 'downward' heuristic with information from
        strawberry detector.
    """

    def __init__(self, action_bound, detector, lut_info, local_p=1.0, 
        thresh=0.5):
        """ Initialize custom policy."""
        super(Policy, self).__init__() 
        self.action_bound = action_bound 
        self.detector = detector
        self.lut_thetas = lut_info['thetas'] 
        self.lut_phis = lut_info['phis'] 
        self.lut_mask = lut_info['mask']  
        self.p = local_p  
        self.thresh = thresh 
        self.name = 'down + stop + in-bounds'

        self.im_ctr = (agent_cfg.obs_shape[0]/2.0, agent_cfg.obs_shape[1]/2.0)
    
    def best_bb(self, bbs):
        """ Returns predicted bounding box for ripe strawberry with highest
            confidence value. 
        """
        best = None 
        max_confidence = 0.0 

        for bb in bbs:
            if bb[0] and (bb[1] > max_confidence):
                max_confidence = bb[1] 
                best = copy.copy(bb)
        
        return best 
    
    def angle_check(self, current_angles, desired_angles):
        """ Determines validity of each angle by checking lut. Returns 
            (bool theta_check, bool phi_check). 
        """
        theta = desired_angles[0] % (2*np.pi) 
        phi = desired_angles[1]
        theta_idx = (np.abs(self.lut_thetas - theta)).argmin() 
        phi_idx = (np.abs(self.lut_phis - phi)).argmin() 

        cur_theta = current_angles[0] % (2*np.pi)
        cur_phi = current_angles[1] 
        cur_theta_idx = (np.abs(self.lut_thetas - cur_theta)).argmin() 
        cur_phi_idx = (np.abs(self.lut_phis - cur_phi)).argmin() 

        if phi < 0.0 or phi > np.pi/2:   
            theta_valid = self.lut_mask[theta_idx, cur_phi_idx]
            phi_valid = False 
        elif self.lut_mask[theta_idx, phi_idx]:
            theta_valid = True 
            phi_valid = True
        elif self.lut_mask[theta_idx, cur_phi_idx]:
            theta_valid = True 
            phi_valid = False 
        elif self.lut_mask[cur_theta_idx, phi_idx]: 
            theta_valid = False 
            phi_valid = True 
        else: 
            theta_valid = False 
            phi_valid = False 

        return (theta_valid, phi_valid)  

    def get_action(self, o, j):
        """ Compute action given observed camera image and position on 
            hemisphere. 
        """
        bgr = cv2.cvtColor(o,  cv2.COLOR_RGB2BGR)
        bbs = self.detector.detect(bgr)
        best_bb = self.best_bb(bbs)
        
        if best_bb is not None: 
            theta_inc = 0.01 * self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = 0.01 * self.action_bound[1] * (2*random.random() - 1.0)
        else: # random theta + down
            theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
            #phi_inc = self.action_bound[1] * (2*random.random() - 1.0)
            phi_inc = self.action_bound[1]

        desired_angles = (j[0] + theta_inc, j[1] + phi_inc)
        current_angles = (j[0], j[1]) 
        theta_valid, phi_valid = \
            self.angle_check(current_angles, desired_angles) 
        
        if not theta_valid: 
            #theta_inc = 0.0 
            theta_inc = -theta_inc 

        if not phi_valid: 
            #phi_inc = 0.0 
            phi_inc = -phi_inc 

        return (theta_inc, phi_inc) 

class CustomPolicy3(Policy):
    """ This custom policy combines 'downward' heuristic with information from
        strawberry detector.
    """

    def __init__(self, action_bound, detector, lut_info, local_p=1.0, 
        thresh=0.5):
        """ Initialize custom policy."""
        super(Policy, self).__init__() 
        self.action_bound = action_bound 
        self.detector = detector
        self.lut_thetas = lut_info['thetas'] 
        self.lut_phis = lut_info['phis'] 
        self.lut_mask = lut_info['mask']  
        self.p = local_p  
        self.thresh = thresh 
        self.name = 'random + in-bounds + stop'

        self.im_ctr = (agent_cfg.obs_shape[0]/2.0, agent_cfg.obs_shape[1]/2.0)
    
    def best_bb(self, bbs):
        """ Returns predicted bounding box for ripe strawberry with highest
            confidence value. 
        """
        best = None 
        max_confidence = 0.0 

        for bb in bbs:
            if bb[0] and (bb[1] > max_confidence):
                max_confidence = bb[1] 
                best = copy.copy(bb)
        
        return best 
    
    def angle_check(self, current_angles, desired_angles):
        """ Determines validity of each angle by checking lut. Returns 
            (bool theta_check, bool phi_check). 
        """
        theta = desired_angles[0] % (2*np.pi) 
        phi = desired_angles[1]
        theta_idx = (np.abs(self.lut_thetas - theta)).argmin() 
        phi_idx = (np.abs(self.lut_phis - phi)).argmin() 

        cur_theta = current_angles[0] % (2*np.pi)
        cur_phi = current_angles[1] 
        cur_theta_idx = (np.abs(self.lut_thetas - cur_theta)).argmin() 
        cur_phi_idx = (np.abs(self.lut_phis - cur_phi)).argmin() 

        if phi < 0.0 or phi > np.pi/2:   
            theta_valid = self.lut_mask[theta_idx, cur_phi_idx]
            phi_valid = False 
        elif self.lut_mask[theta_idx, phi_idx]:
            theta_valid = True 
            phi_valid = True
        elif self.lut_mask[theta_idx, cur_phi_idx]:
            theta_valid = True 
            phi_valid = False 
        elif self.lut_mask[cur_theta_idx, phi_idx]: 
            theta_valid = False 
            phi_valid = True 
        else: 
            theta_valid = False 
            phi_valid = False 

        return (theta_valid, phi_valid)  

    def get_action(self, o, j):
        """ Compute action given observed camera image and position on 
            hemisphere. 
        """
        bgr = cv2.cvtColor(o,  cv2.COLOR_RGB2BGR)
        bbs = self.detector.detect(bgr)
        best_bb = self.best_bb(bbs)
        
        if best_bb is not None: 
            theta_inc = 0.01 * self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = 0.01 * self.action_bound[1] * (2*random.random() - 1.0)
        else: # random 
            theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = self.action_bound[1] * (2*random.random() - 1.0)

        desired_angles = (j[0] + theta_inc, j[1] + phi_inc)
        current_angles = (j[0], j[1]) 
        theta_valid, phi_valid = \
            self.angle_check(current_angles, desired_angles) 
        
        if not theta_valid: 
            #theta_inc = 0.0 
            theta_inc = -theta_inc 

        if not phi_valid: 
            #phi_inc = 0.0 
            phi_inc = -phi_inc 

        return (theta_inc, phi_inc) 

class CustomPolicy4(Policy):
    """ This custom policy combines 'downward' heuristic with information from
        strawberry detector.
    """

    def __init__(self, action_bound, detector, lut_info, local_p=1.0, 
        thresh=0.5):
        """ Initialize custom policy."""
        super(Policy, self).__init__() 
        self.action_bound = action_bound 
        self.detector = detector
        self.lut_thetas = lut_info['thetas'] 
        self.lut_phis = lut_info['phis'] 
        self.lut_mask = lut_info['mask']  
        self.p = local_p  
        self.thresh = thresh 
        self.name = 'random + in-bounds'

        self.im_ctr = (agent_cfg.obs_shape[0]/2.0, agent_cfg.obs_shape[1]/2.0)
    
    def best_bb(self, bbs):
        """ Returns predicted bounding box for ripe strawberry with highest
            confidence value. 
        """
        best = None 
        max_confidence = 0.0 

        for bb in bbs:
            if bb[0] and (bb[1] > max_confidence):
                max_confidence = bb[1] 
                best = copy.copy(bb)
        
        return best 
    
    def angle_check(self, current_angles, desired_angles):
        """ Determines validity of each angle by checking lut. Returns 
            (bool theta_check, bool phi_check). 
        """
        theta = desired_angles[0] % (2*np.pi) 
        phi = desired_angles[1]
        theta_idx = (np.abs(self.lut_thetas - theta)).argmin() 
        phi_idx = (np.abs(self.lut_phis - phi)).argmin() 

        cur_theta = current_angles[0] % (2*np.pi)
        cur_phi = current_angles[1] 
        cur_theta_idx = (np.abs(self.lut_thetas - cur_theta)).argmin() 
        cur_phi_idx = (np.abs(self.lut_phis - cur_phi)).argmin() 

        if phi < 0.0 or phi > np.pi/2:   
            theta_valid = self.lut_mask[theta_idx, cur_phi_idx]
            phi_valid = False 
        elif self.lut_mask[theta_idx, phi_idx]:
            theta_valid = True 
            phi_valid = True
        elif self.lut_mask[theta_idx, cur_phi_idx]:
            theta_valid = True 
            phi_valid = False 
        elif self.lut_mask[cur_theta_idx, phi_idx]: 
            theta_valid = False 
            phi_valid = True 
        else: 
            theta_valid = False 
            phi_valid = False 

        return (theta_valid, phi_valid)  

    def get_action(self, o, j):
        """ Compute action given observed camera image and position on 
            hemisphere. 
        """
        # Random + try to stay in bounds
        time.sleep(1.0)
        theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
        phi_inc = self.action_bound[1] * (2*random.random() - 1.0)

        desired_angles = (j[0] + theta_inc, j[1] + phi_inc)
        current_angles = (j[0], j[1]) 
        theta_valid, phi_valid = \
            self.angle_check(current_angles, desired_angles) 
        
        if not theta_valid: 
            #theta_inc = 0.0 
            theta_inc = -theta_inc 

        if not phi_valid: 
            #phi_inc = 0.0 
            phi_inc = -phi_inc 

        return (theta_inc, phi_inc) 

class DDPGPolicy(Policy):
    """ Object to run DDPG actor policy. """

    def __init__(self, session, actor, weights_file, name=None):
        """ Initialize DDPG actor policy. """
        super(Policy, self).__init__()
        self.actor = actor
        self.session = session
        self.weights_file = weights_file
        if name is None:
            self.name = 'ddpg'
        else:
            self.name = name 

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        self.session.run(tf.global_variables_initializer())
        self.restore_weights()

    def get_action(self, o, j):
        """ Computes action given observed camera image and position on
            hemisphere.
        """
        o_reshape = np.reshape(o, (1, self.actor.obs_shape[0],
            self.actor.obs_shape[1],
            self.actor.obs_shape[2]))
        j_reshape = np.reshape(j, (1, self.actor.pos_shape[0]))
        return self.actor.predict(o_reshape, j_reshape, batch_size=1,
            training=0, add_noise=False)

    def restore_weights(self):
        """ Restores weights from saved location. """
        print('{} Restoring weights from: {}...'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
            self.weights_file))
        self.saver.restore(self.session, self.weights_file)
        print('Done.')
        sys.stdout.flush()

class CustomPolicy5(Policy):
    """ This custom policy combines 'downward' heuristic with information from
        strawberry detector.
    """

    def __init__(self, action_bound, detector, lut_info, phi_thresh=np.pi/4):
        """ Initialize custom policy."""
        super(Policy, self).__init__() 
        self.action_bound = action_bound 
        self.detector = detector
        self.lut_thetas = lut_info['thetas'] 
        self.lut_phis = lut_info['phis'] 
        self.lut_mask = lut_info['mask']  
        self.phi_thresh = phi_thresh
        self.name = 'down + stop + in-bounds 2, t6'

    def best_bb(self, bbs):
        """ Returns predicted bounding box for ripe strawberry with highest
            confidence value. 
        """
        best = None 
        max_confidence = 0.6

        for bb in bbs:
            if bb[0] and (bb[1] > max_confidence):
                max_confidence = bb[1] 
                best = copy.copy(bb)
        
        return best 
    
    def angle_check(self, current_angles, desired_angles):
        """ Determines validity of each angle by checking lut. Returns 
            (bool theta_check, bool phi_check). 
        """
        theta = desired_angles[0] % (2*np.pi) 
        phi = desired_angles[1]
        theta_idx = (np.abs(self.lut_thetas - theta)).argmin() 
        phi_idx = (np.abs(self.lut_phis - phi)).argmin() 

        cur_theta = current_angles[0] % (2*np.pi)
        cur_phi = current_angles[1] 
        cur_theta_idx = (np.abs(self.lut_thetas - cur_theta)).argmin() 
        cur_phi_idx = (np.abs(self.lut_phis - cur_phi)).argmin() 

        if phi < 0.0 or phi > np.pi/2:   
            theta_valid = self.lut_mask[theta_idx, cur_phi_idx]
            phi_valid = False 
        elif self.lut_mask[theta_idx, phi_idx]:
            theta_valid = True 
            phi_valid = True
        elif self.lut_mask[theta_idx, cur_phi_idx]:
            theta_valid = True 
            phi_valid = False 
        elif self.lut_mask[cur_theta_idx, phi_idx]: 
            theta_valid = False 
            phi_valid = True 
        else: 
            theta_valid = False 
            phi_valid = False 

        return (theta_valid, phi_valid)  

    def get_action(self, o, j):
        """ Compute action given observed camera image and position on 
            hemisphere. 
        """
        bgr = cv2.cvtColor(o,  cv2.COLOR_RGB2BGR)
        bbs = self.detector.detect(bgr)
        best_bb = self.best_bb(bbs)
        
        time.sleep(0.25)

        if j[1] < self.phi_thresh: # down + random theta
            theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = self.action_bound[1]
        elif best_bb is not None: # stop
            theta_inc = 0.001 * self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = 0.001 * self.action_bound[1] * (2*random.random() - 1.0)
        else: # random
            theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = self.action_bound[1] * (2*random.random() - 1.0)

        desired_angles = (j[0] + theta_inc, j[1] + phi_inc)
        current_angles = (j[0], j[1]) 
        theta_valid, phi_valid = \
            self.angle_check(current_angles, desired_angles) 
        
        if not theta_valid: 
            theta_inc = -theta_inc 

        if not phi_valid: 
            phi_inc = -phi_inc 

        return (theta_inc, phi_inc) 

class CustomPolicy6(Policy):
    """ This custom policy combines 'downward' heuristic with information from
        strawberry detector.
    """

    def __init__(self, action_bound, detector, lut_info, local_p=1.0, 
        phi_thresh=np.pi/4, thresh=0.5):
        """ Initialize custom policy."""
        super(Policy, self).__init__() 
        self.action_bound = action_bound 
        self.detector = detector
        self.lut_thetas = lut_info['thetas'] 
        self.lut_phis = lut_info['phis'] 
        self.lut_mask = lut_info['mask']  
        self.p = local_p  
        self.thresh = thresh 
        self.phi_thresh = phi_thresh
        self.name = 'down + proportional + in-bounds 2, t6'

        self.im_ctr = (agent_cfg.obs_shape[0]/2.0, agent_cfg.obs_shape[1]/2.0)
    
    def best_bb(self, bbs):
        """ Returns predicted bounding box for ripe strawberry with highest
            confidence value. 
        """
        best = None 
        max_confidence = 0.6 

        for bb in bbs:
            if bb[0] and (bb[1] > max_confidence):
                max_confidence = bb[1] 
                best = copy.copy(bb)
        
        return best 
    
    def angle_check(self, current_angles, desired_angles):
        """ Determines validity of each angle by checking lut. Returns 
            (bool theta_check, bool phi_check). 
        """
        theta = desired_angles[0] % (2*np.pi) 
        phi = desired_angles[1]
        theta_idx = (np.abs(self.lut_thetas - theta)).argmin() 
        phi_idx = (np.abs(self.lut_phis - phi)).argmin() 

        cur_theta = current_angles[0] % (2*np.pi)
        cur_phi = current_angles[1] 
        cur_theta_idx = (np.abs(self.lut_thetas - cur_theta)).argmin() 
        cur_phi_idx = (np.abs(self.lut_phis - cur_phi)).argmin() 

        if phi < 0.0 or phi > np.pi/2:   
            theta_valid = self.lut_mask[theta_idx, cur_phi_idx]
            phi_valid = False 
        elif self.lut_mask[theta_idx, phi_idx]:
            theta_valid = True 
            phi_valid = True
        elif self.lut_mask[theta_idx, cur_phi_idx]:
            theta_valid = True 
            phi_valid = False 
        elif self.lut_mask[cur_theta_idx, phi_idx]: 
            theta_valid = False 
            phi_valid = True 
        else: 
            theta_valid = False 
            phi_valid = False 

        return (theta_valid, phi_valid)  

    def get_action(self, o, j):
        """ Compute action given observed camera image and position on 
            hemisphere. 
        """
        bgr = cv2.cvtColor(o,  cv2.COLOR_RGB2BGR)
        bbs = self.detector.detect(bgr)
        best_bb = self.best_bb(bbs)
        
        time.sleep(0.25)

        if j[1] < self.phi_thresh: # down + random theta
            theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = self.action_bound[1]
        elif best_bb is not None: # proportional control
            x_frac = (best_bb[2][1] - self.im_ctr[0]) / self.im_ctr[0]
            y_frac = (best_bb[2][0] - self.im_ctr[1]) / self.im_ctr[1] 
            
            if abs(x_frac) > self.thresh:
                phi_inc = np.sign(x_frac) * self.action_bound[1] 
            else: 
                phi_inc = self.p * x_frac * self.action_bound[1]
            
            if abs(y_frac) > self.thresh: 
                theta_inc = np.sign(y_frac) * self.action_bound[0] 
            else: 
                theta_inc = self.p * y_frac * self.action_bound[1] 
        else: # random
            theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = self.action_bound[1] * (2*random.random() - 1.0)

        desired_angles = (j[0] + theta_inc, j[1] + phi_inc)
        current_angles = (j[0], j[1]) 
        theta_valid, phi_valid = \
            self.angle_check(current_angles, desired_angles) 
        
        if not theta_valid: 
            theta_inc = -theta_inc 

        if not phi_valid: 
            phi_inc = -phi_inc 

        return (theta_inc, phi_inc) 

class CustomPolicy7(Policy):
    """ This custom policy combines 'downward' heuristic location info.
    """

    def __init__(self, action_bound, detector, lut_info, local_p=1.0, 
        thresh=0.5, phi_thresh=np.pi/4):
        """ Initialize custom policy."""
        super(Policy, self).__init__() 
        self.action_bound = action_bound 
        self.detector = detector
        self.lut_thetas = lut_info['thetas'] 
        self.lut_phis = lut_info['phis'] 
        self.lut_mask = lut_info['mask']  
        self.p = local_p  
        self.thresh = thresh 
        self.phi_thresh = phi_thresh
        self.name = 'random + in-bounds 2'

        self.im_ctr = (agent_cfg.obs_shape[0]/2.0, agent_cfg.obs_shape[1]/2.0)
    
    def best_bb(self, bbs):
        """ Returns predicted bounding box for ripe strawberry with highest
            confidence value. 
        """
        best = None 
        max_confidence = 0.0 

        for bb in bbs:
            if bb[0] and (bb[1] > max_confidence):
                max_confidence = bb[1] 
                best = copy.copy(bb)
        
        return best 
    
    def angle_check(self, current_angles, desired_angles):
        """ Determines validity of each angle by checking lut. Returns 
            (bool theta_check, bool phi_check). 
        """
        theta = desired_angles[0] % (2*np.pi) 
        phi = desired_angles[1]
        theta_idx = (np.abs(self.lut_thetas - theta)).argmin() 
        phi_idx = (np.abs(self.lut_phis - phi)).argmin() 

        cur_theta = current_angles[0] % (2*np.pi)
        cur_phi = current_angles[1] 
        cur_theta_idx = (np.abs(self.lut_thetas - cur_theta)).argmin() 
        cur_phi_idx = (np.abs(self.lut_phis - cur_phi)).argmin() 

        if phi < 0.0 or phi > np.pi/2:   
            theta_valid = self.lut_mask[theta_idx, cur_phi_idx]
            phi_valid = False 
        elif self.lut_mask[theta_idx, phi_idx]:
            theta_valid = True 
            phi_valid = True
        elif self.lut_mask[theta_idx, cur_phi_idx]:
            theta_valid = True 
            phi_valid = False 
        elif self.lut_mask[cur_theta_idx, phi_idx]: 
            theta_valid = False 
            phi_valid = True 
        else: 
            theta_valid = False 
            phi_valid = False 

        return (theta_valid, phi_valid)  

    def get_action(self, o, j):
        """ Compute action given observed camera image and position on 
            hemisphere. 
        """
        # Random + try to stay in bounds

        time.sleep(1.0)
        if j[1] < self.phi_thresh: # down + random theta
            theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = self.action_bound[1]
        else: # random
            theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = self.action_bound[1] * (2*random.random() - 1.0)

        desired_angles = (j[0] + theta_inc, j[1] + phi_inc)
        current_angles = (j[0], j[1]) 
        theta_valid, phi_valid = \
            self.angle_check(current_angles, desired_angles) 
        
        if not theta_valid: 
            #theta_inc = 0.0 
            theta_inc = -theta_inc 

        if not phi_valid: 
            #phi_inc = 0.0 
            phi_inc = -phi_inc 

        return (theta_inc, phi_inc) 

class HybridPolicy(Policy):
    """ Class for running heuristic + DDPG hybrid policy. """

    def __init__(self, session, actor, weights_file, action_bound, lut_info, 
        phi_thresh=np.pi/4, name=None):
        """ Initialize DDPG actor policy. """
        super(Policy, self).__init__()
        self.actor = actor
        self.session = session
        self.weights_file = weights_file
        if name is None:
            self.name = 'ddpg + down + in-bounds'
        else:
            self.name = name 

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        self.session.run(tf.global_variables_initializer())
        self.restore_weights()
        
        self.action_bound = action_bound
        self.phi_thresh = phi_thresh 
        self.lut_thetas = lut_info['thetas'] 
        self.lut_phis = lut_info['phis'] 
        self.lut_mask = lut_info['mask']  

    def _ddpg_action(self, o, j):
        o_reshape = np.reshape(o, (1, self.actor.obs_shape[0],
            self.actor.obs_shape[1],
            self.actor.obs_shape[2]))
        j_reshape = np.reshape(j, (1, self.actor.pos_shape[0]))
        return self.actor.predict(o_reshape, j_reshape, batch_size=1,
            training=0, add_noise=False)

    def angle_check(self, current_angles, desired_angles):
        """ Determines validity of each angle by checking lut. Returns 
            (bool theta_check, bool phi_check). 
        """
        theta = desired_angles[0] % (2*np.pi) 
        phi = desired_angles[1]
        theta_idx = (np.abs(self.lut_thetas - theta)).argmin() 
        phi_idx = (np.abs(self.lut_phis - phi)).argmin() 

        cur_theta = current_angles[0] % (2*np.pi)
        cur_phi = current_angles[1] 
        cur_theta_idx = (np.abs(self.lut_thetas - cur_theta)).argmin() 
        cur_phi_idx = (np.abs(self.lut_phis - cur_phi)).argmin() 

        if phi < 0.0 or phi > np.pi/2:   
            theta_valid = self.lut_mask[theta_idx, cur_phi_idx]
            phi_valid = False 
        elif self.lut_mask[theta_idx, phi_idx]:
            theta_valid = True 
            phi_valid = True
        elif self.lut_mask[theta_idx, cur_phi_idx]:
            theta_valid = True 
            phi_valid = False 
        elif self.lut_mask[cur_theta_idx, phi_idx]: 
            theta_valid = False 
            phi_valid = True 
        else: 
            theta_valid = False 
            phi_valid = False 

        return (theta_valid, phi_valid)  

    def get_action(self, o, j):
        """ Computes action given observed camera image and position on
            hemisphere.
        """
        
        if j[1] < self.phi_thresh:
            time.sleep(1.0)
            theta_inc = self.action_bound[0] * (2*random.random() - 1.0)
            phi_inc = self.action_bound[1]
        else:
            a_array = self._ddpg_action(o, j)
            theta_inc = a_array[0][0]
            phi_inc = a_array[0][1]
        
        desired_angles = (j[0] + theta_inc, j[1] + phi_inc)
        current_angles = (j[0], j[1]) 
        theta_valid, phi_valid = \
            self.angle_check(current_angles, desired_angles) 
        
        if not theta_valid: 
            theta_inc = -theta_inc 

        if not phi_valid: 
            phi_inc = -phi_inc 

        return (theta_inc, phi_inc)         

    def restore_weights(self):
        """ Restores weights from saved location. """
        print('{} Restoring weights from: {}...'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
            self.weights_file))
        self.saver.restore(self.session, self.weights_file)
        print('Done.')
        sys.stdout.flush()

class RandomPolicy(Policy):
    """ Object to run random actor policy. """

    def __init__(self, action_bound):
        """ Initialize random policy. """
        super(Policy, self).__init__()
        self.action_bound = action_bound
        self.name = 'random'

    def get_action(self, o, j):
        """ Outputs random action, irrespective of what is the input state. """
        time.sleep(1.0) # Make sure trajectory planner does not use old states (likely overkill)
        unscaled = 2*np.random.rand(*self.action_bound.shape) - 1.0
        return self.action_bound*unscaled

class Evaluator(object):
    """ Object for evaluating policies. """

    def __init__(self, policies, output_dir, agent, results_file=None, 
        session=None, episode_length=100, num_episodes=100):
        """ Initialize evaluator object. """
        self.agent = agent
        self.policies = policies
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        self.session = session 
        self.results_file = results_file
        self.test_results = {}

        self.output_dir = output_dir
        self.summary_dir = os.path.join(self.output_dir,
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if self.results_file:
            self.load_results(self.results_file)

        self.save_config()

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def save_config(self):
        """ Saves testing configuration. """
        with open(os.path.join(self.summary_dir, 'config.txt'), 'w') as f:
            f.write('episode length: ' + str(self.episode_length) + '\n')
            f.write('num episodes: ' + str(self.num_episodes) + '\n')
            f.write('results file: ' + str(self.results_file) + '\n')
            for policy in self.policies:
                f.write('name: ' + policy.name + '\n')
                if policy.name == 'ddpg':
                    f.write('  weights file: ' + policy.weights_file + '\n')
    
    # def save_testing_variables(self):
    #     """ Saves testing variables in summary directory. """
    #     print('{} Saving testing variables to: {}'.format(
    #         datetime.datetime.now().strftime('%m-%d %H:%M:%S'), 
    #         self.summary_dir))
    #     var_dict = {
    #         ''
    #     }
    #     save_loc = os.path.join(self.summary_dir, 'testing_vars.pkl')
    #     with open(save_loc, 'w') as f:
    #         pickle.dump(var_dict, f)
    #     print('Done.')
    #     sys.stdout.flush()

    def save_results(self):
        """ Saves results stored in memberdata to pkl file. """
        print('{} Saving results to: {}...'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
            self.summary_dir))
        save_loc = os.path.join(self.summary_dir, 'test_results.pkl')
        with open(save_loc, 'w') as f:
            pickle.dump(self.test_results, f)
        print('Done.')
        sys.stdout.flush()
    
    def load_results(self, results_file):
        """ Loads results from pkl file to memberdata. """
        print('{} Loading results from: {}...'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
            results_file))    
        with open(results_file, 'r') as f:
            self.test_results = pickle.load(f)
        print('Done.')
        sys.stdout.flush()

    def test_policy(self, policy):
        """ Tests policy and saves data in summary dir. """
        print('Testing policy: ' + policy.name)
        ep_ave_rewards = []
        ep_term_rewards = []
        ep_max_rewards = []
        ep_total_rewards = []
        ep_length = []

        for ep in range(self.num_episodes):
            print('Starting episode ' + str(ep))
            ep_rewards = []

            o, j = self.agent.reset()

            for step in range(self.episode_length):
                a = policy.get_action(o, j)
                [o, j], r, t, _ = self.agent.step(a)
                ep_rewards.append(r)

                if t:
                    break

            print('Episode ' + str(ep) + ' complete. Updating statistics.')
            ep_max_rewards.append(max(ep_rewards))
            ep_ave_rewards.append(np.mean(ep_rewards))
            ep_term_rewards.append(ep_rewards[-1])
            ep_total_rewards.append(sum(ep_rewards))
            ep_length.append(step+1)

        self.test_results[policy.name] = {
            'max rewards': ep_max_rewards,
            'average rewards': ep_ave_rewards,
            'terminal rewards': ep_term_rewards, 
            'total rewards': ep_total_rewards,
            'episode lengths': ep_length}
        print('Test complete. Saving updated results.')
        self.save_results()

    def compare_policies(self):
        """ Runs policy evaluation on each policy in memberdata and saves
            results.
        """
        for policy in self.policies:
            if self.test_results.get(policy.name) is None:
                self.test_policy(policy)
        self.display_results(self.test_results, thresh=0)
        self.display_results(self.test_results, thresh=4)

    
    def display_results(self, results, thresh=0):
        """ Summarizes results stored in memberdata. """
        import matplotlib.pyplot as plt 

        # Rearrange data for plotting
        plot_data = {} 

        for policy, stats in results.iteritems():
            save = [x[0] for x in enumerate(stats['episode lengths']) 
                if x[1] > thresh]
            if thresh != 0:
                if plot_data.get('suicides') is None:
                    plot_data['suicides'] = {}
                plot_data['suicides'][policy] = \
                    len(stats['episode lengths']) - len(save)

            for metric, values in stats.iteritems():       
                pruned = [values[i] for i in save]         
                val = float(sum(pruned))/len(pruned)
                
                if plot_data.get(metric) is None:
                    plot_data[metric] = {}

                plot_data[metric][policy] = val
        
        fig_num = 0
        for metric, policies in plot_data.iteritems():
            plt.figure(fig_num)
            plt.bar(range(len(policies)), list(policies.values()), 
                align='center')
            plt.xticks(range(len(policies)), list(policies.keys()))
            if thresh != 0:
                plt.title(metric + ' (thresh = ' + str(thresh) + ')')
            else:
                plt.title(metric)
            fig_num += 1
        
        plt.show()
    
    def print_results(self, results):
        """ Summarizes results stored in memberdata. """
        for policy, stats in results.iteritems():
            print(str(policy))
            for metric, values in stats.iteritems():
                val = float(sum(values))/len(values)
                print(' ' + str(metric) + ': ' + str(val) + 
                    ' (averaged over ' + str(len(values)) + 
                    ' episodes)')
        
    def exit_gracefully(self, sig, frame):
        """ Saves results and closes tensorflow session before exiting. """
        print('Signal: ' + str(sig))
        self.save_results()
        try:
            self.session.close()
        except:
            pass 

        kill_named_processes(name='roslaunch', sig=signal.SIGTERM)
        time.sleep(2)
        kill_named_processes(name='roscore', sig=signal.SIGTERM)
        time.sleep(2)

        kill_named_processes(name='roslaunch', sig=signal.SIGKILL)   
        kill_named_processes(name='move_group', sig=signal.SIGKILL)  
        kill_named_processes(name='robot_state_publisher', sig=signal.SIGKILL)       
        kill_named_processes(name='gzserver', sig=signal.SIGKILL) 
        kill_named_processes(name='gzclient', sig=signal.SIGKILL)   
        kill_named_processes(name='roscore',  sig=signal.SIGKILL)

        sys.exit()

def main(args_dict):
    config = tf.ConfigProto(**ddpg_cfg.tf_cfg)
    config.gpu_options.allow_growth = True 
    with tf.Session(config=config) as session:
        np.random.seed(ddpg_cfg.np_seed)
        tf.set_random_seed(ddpg_cfg.tf_seed)

        [obs_shape, action_shape] = agent_cfg.hemi_state_shape
        action_bound = agent_cfg.hemi_action_bound
        OU_noise = noise.OrnsteinUhlenbeckActionNoise(
            mu=agent_cfg.mu,
            sigma=agent_cfg.sigma,
            theta=agent_cfg.theta)

        # initialize function approximators
        # embedding_network = networks.EmbeddingNetwork(session)
        embedding_network = None
        actor_network = networks.ActorNetwork(
            session,
            obs_shape,
            action_shape,
            action_bound,
            OU_noise,
            embedding=embedding_network)

        agent = agent_ros.HemiAgentROS(headless=True, feed=False, detector=True)
        # agent = agent_ros.HemiAgentROS(headless=False, feed=True, detector=True)
        lut_info = {'thetas': agent.lut_thetas, 'phis': agent.lut_phis, 
            'mask': agent.lut_mask}

        # ddpg_policy = DDPGPolicy(session=session, actor=actor_network,
        #     weights_file=args_dict['weights_file'])
        # expert_policy1 = CustomPolicy1(action_bound=action_bound, 
        #     detector=agent.detector_feedback.detector, lut_info=lut_info)
        # expert_policy2 = CustomPolicy2(action_bound=action_bound, 
        #     detector=agent.detector_feedback.detector, lut_info=lut_info)
        # expert_policy3 = CustomPolicy3(action_bound=action_bound, 
        #     detector=agent.detector_feedback.detector, lut_info=lut_info)
        # expert_policy4 = CustomPolicy4(action_bound=action_bound, 
        #     detector=agent.detector_feedback.detector, lut_info=lut_info)
        # random_policy = RandomPolicy(action_bound=action_bound)

        hybrid_base = 'ddpg + down + in-bounds'
        # hybrid_policy1 = HybridPolicy(session=session, actor=actor_network,
        #     weights_file=args_dict['weights_file'], action_bound=action_bound,
        #     lut_info=lut_info, phi_thresh=np.pi/6, name=hybrid_base+', thresh=pi/6')
        # hybrid_policy2 = HybridPolicy(session=session, actor=actor_network,
        #     weights_file=args_dict['weights_file'], action_bound=action_bound,
        #     lut_info=lut_info, phi_thresh=np.pi/4, name=hybrid_base+', thresh=pi/4')
        # hybrid_policy3 = HybridPolicy(session=session, actor=actor_network,
        #     weights_file=args_dict['weights_file'], action_bound=action_bound,
        #     lut_info=lut_info, phi_thresh=np.pi/3, name=hybrid_base+', thresh=pi/3')

        hybrid_policy = HybridPolicy(session=session, actor=actor_network,
            weights_file=args_dict['weights_file'], action_bound=action_bound,
            lut_info=lut_info, phi_thresh=np.pi/4, name=hybrid_base)
        # policies = [ddpg_policy, expert_policy1, expert_policy2, 
        #     expert_policy3, expert_policy4, random_policy]
        # policies = [ddpg_policy, expert_policy1, random_policy]
        expert_policy5 = CustomPolicy5(action_bound=action_bound, 
            detector=agent.detector_feedback.detector, lut_info=lut_info,
            phi_thresh=np.pi/4)
        expert_policy6 = CustomPolicy6(action_bound=action_bound, 
            detector=agent.detector_feedback.detector, lut_info=lut_info,
            phi_thresh=np.pi/4)            
        expert_policy7 = CustomPolicy7(action_bound=action_bound,
            detector=agent.detector_feedback.detector,lut_info=lut_info, 
            phi_thresh=np.pi/4)

        policies = [expert_policy5, expert_policy6, expert_policy7]
        evaluator = Evaluator(policies=policies, agent=agent, 
            output_dir=args_dict['output_dir'], session=session,
            results_file=args_dict['results_file'], 
            episode_length=agent.max_episode_steps)

        evaluator.compare_policies()
        print('Comparison complete. Terminating program.')
        os.kill(os.getpid(), signal.SIGTERM)

if __name__ == '__main__':
    # Parse command-line argumnets and run algorithm
    parser = argparse.ArgumentParser(
        description='provide arguments for policy evaluation')
    parser.add_argument('--continue-test',
        help='continue testing with data from previous trials',
        default=False,
        action='store_true')
    parser.add_argument('--output-dir',
        help='directory for logging test info and results',
        default='/mnt/storage/testing')
    parser.add_argument('--results-file',
        help='file containing test results from current trial',
        default='',
        type=str)
    parser.add_argument('--weights-file',
        help='file containing pretrained weights (leave empty to get latest)',
        default='',
        type=str)
    args_dict = vars(parser.parse_args())
    
    if args_dict['continue_test']:
        args_dict['weights_file'], args_dict['results_file'] = \
            get_latest(args_dict['output_dir'])
    elif args_dict['weights_file'] == '':
        args_dict['weights_file'] = get_latest_weights(args_dict['output_dir'])
    main(args_dict)

    # python policy.py --results-file '/mnt/storage/testing/2018_10_22_07_55/test_results.pkl' --weights-file '/mnt/storage/results/2018_10_21_10_20/ddpg'
    # python policy.py --weights-file '/mnt/storage/results/2018_10_21_10_20/ddpg'
# python policy.py --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'