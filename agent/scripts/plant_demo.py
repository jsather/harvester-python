""" plant_demo.py generates different plant configurations for demonstration
    purposes.
"""
import time 

import numpy as np 

import agent.plant_ros as plant_ros 
import agent.agent_utils as agent_utils 

import pdb 

def plants_in_row(plant, names=['model']):
    spacing = (2/np.sqrt(2)) * abs(plant.bed_dy)/(plant.num_rows + 1)
    x = plant.bed_corner[0] + np.sign(plant.bed_dx)*spacing*1.5#np.sign(plant.bed_dx)*spacing/2

    for plant_num, name in enumerate(names):
        plant_name = 'plant' + str(plant_num)
        plant.new(x=x, plant_model_prefix=name, remove=False, name=plant_name)
        x = x + np.sign(plant.bed_dx)*spacing
        time.sleep(15)

def main():
    plant_names = ['model_mu20_sd0', 'model_mu275_sd0', 'model_mu35_sd0', 
        'model_mu425_sd0', 'model_mu50_sd0']
    plant_names.reverse() 

    sim = agent_utils.HarvesterSimulation() 
    sim.cfg['paused'] = False 
    sim.run(world_only=True) 

    plant = plant_ros.PlantROS() 
    plants_in_row(plant, plant_names) 
    pdb.set_trace()

if __name__ == "__main__":
    main()