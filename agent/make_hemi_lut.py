""" Script to construct look up table for HemiAgentROS. It's gross, I know...
    But it works!
    Author: Jonathon Sather 
    Last updated: idk/2018
"""
from time import sleep
import numpy as np
import os
import sys
import pickle
import copy 

import agent_ros
import plant_ros
import utils as agent_utils

save_loc = os.path.join(os.path.expanduser('~'), 'projects', 'harvester',
    'python', 'agent', 'joint_angles', 'lut_final3')

def fix_lut(lut, thetas, phis):
    NUM_THETA = len(thetas)
    NUM_PHI = len(phis)

    valid_angles = np.zeros((NUM_THETA, NUM_PHI))
    summary = {}

    for t in range(NUM_THETA):
        for p in range(NUM_PHI):
            if np.isnan(lut[t, p , 0]):
                valid_angles[t, p] = 0
                if p == 0:
                    lut[t, p, :] = lut[t, p+1, :]
                else:
                    lut[t, p:, :] = lut[t, p-1, :]
            else:
                valid_angles[t, p] = 1

    summary['lut'] = lut
    summary['thetas'] = thetas
    summary['phis'] = phis
    summary['valid_angles'] = valid_angles
    with open(save_loc, 'w') as f:
        pickle.dump(summary, f)
    print("Look up table complete! Saved in " + save_loc)

def main():
    """ Collect hemi data and store in lookup table. """
    NUM_THETA = 360
    NUM_PHI = 90
    NUM_JOINTS = 6

    agent = agent_ros.HemiAgentROS()
    thetas = np.linspace(0.0001, 2*np.pi-0.0001, NUM_THETA)
    phis = np.linspace(0.0001, np.pi/2, NUM_PHI)
    lut = np.empty((NUM_THETA, NUM_PHI, NUM_JOINTS))
    lut[:] = np.nan
    valid_angles = np.zeros((NUM_THETA, NUM_PHI))
    summary = {}

    sleep(5)
    seed_angles = list(agent.get_joint_angles())
    top_seed_angles = copy.copy(seed_angles)
    og_seed_angles = copy.copy(seed_angles)

    print("Constructing look up table with num theta = " + str(NUM_THETA) +
        " and num phi = " + str(NUM_PHI))
    print("CW")
    # cw from 0 until at base
    for t in range(NUM_THETA-1, int(0.75*NUM_THETA-1), -1):
        num_bad_phi = 0
        for p in range(NUM_PHI):
            print("Computing angle theta: " + str(thetas[t]) + ", phi: "
                + str(phis[p]))
            theta = (thetas[t] + np.pi) % (2*np.pi) - np.pi
            phi = phis[p]

            pose = agent._angles_to_pose(theta, phi)
            ps = agent.stamp_pose(pose)
            if p == 0:
                req = agent.get_ik(pose=ps,seed_angles=top_seed_angles, ik_attempts=1)
            else:
                req = agent.get_ik(pose=ps,seed_angles=seed_angles, ik_attempts=1)
                # req = agent.get_ik(pose=ps, ik_attempts=2)
            if req.error_code.val == 1:
                print("Solution found")
                joint_angles = np.array(req.solution.joint_state.position[:-2])
                joint_angles[[0,3,5]] = (joint_angles[[0,3,5]] + np.pi) % (2*np.pi) - np.pi
                lut[t, p, :] = joint_angles
                valid_angles[t, p] = 1
                # agent.execute_joint_trajectory(joint_angles)
                if p == 0:
                    print("Updating top seed at theta number: " + str(t) +
                        ", phi number: " + str(p))
                    seed_angles = list(joint_angles[:])
                    top_seed_angles = copy.copy(seed_angles)
                else:
                    print("Updating seed at theta number: " + str(t) +
                        ", phi number: " + str(p))
                    seed_angles = list(joint_angles[:])
            else:
                print("No solution found at theta number: " + str(t) +
                    ", phi number: " + str(p))
                num_bad_phi+=1
                if num_bad_phi == 1:
                    lut[t, p, :] = lut[t, p-1, :]
                else:
                    lut[t, p:, :] = lut[t, p-1, :]
                    break

    seed_angles = copy.copy(og_seed_angles)
    top_seed_angles = copy.copy(seed_angles)

    # ccw from 0 until at base
    print("CCW")
    for t in range(int(0.75*NUM_THETA)):
        num_bad_phi = 0
        for p in range(NUM_PHI):
            print("Computing angle theta: " + str(thetas[t]) + ", phi: "
                + str(phis[p]))
            theta = (thetas[t] + np.pi) % (2*np.pi) - np.pi
            phi = phis[p]

            pose = agent._angles_to_pose(theta, phi)
            ps = agent.stamp_pose(pose)
            if p == 0:
                req = agent.get_ik(pose=ps,seed_angles=top_seed_angles, ik_attempts=2)
            else:
                req = agent.get_ik(pose=ps,seed_angles=seed_angles, ik_attempts=2)
                # req = agent.get_ik(pose=ps, ik_attempts=2)
            if req.error_code.val == 1:
                print("Solution found")
                joint_angles = np.array(req.solution.joint_state.position[:-2])
                joint_angles[[0,3,5]] = (joint_angles[[0,3,5]] + np.pi) % (2*np.pi) - np.pi # see if can get away without this!
                lut[t, p, :] = joint_angles
                valid_angles[t, p] = 1
                # agent.execute_joint_trajectory(joint_angles)
                if p == 0:
                    print("Updating top seed at theta number: " + str(t) +
                        ", phi number: " + str(p))
                    seed_angles = list(joint_angles[:])
                    top_seed_angles = copy.copy(seed_angles)
                else:
                    print("Updating seed at theta number: " + str(t) +
                        ", phi number: " + str(p))
                    seed_angles = list(joint_angles[:])
            else:
                print("No solution found at theta number: " + str(t) +
                    ", phi number: " + str(p))
                num_bad_phi+=1
                if num_bad_phi == 1:
                    lut[t, p, :] = lut[t, p-1, :]
                else:
                    lut[t, p:, :] = lut[t, p-1, :]
                    break
    summary['lut'] = lut
    summary['thetas'] = thetas
    summary['phis'] = phis
    summary['valid_angles'] = valid_angles
    with open(save_loc, 'w') as f:
        pickle.dump(summary, f)
    print("Look up table complete! Saved in " + save_loc)

if __name__ == '__main__':
    main()
