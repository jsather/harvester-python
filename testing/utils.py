""" utils.py contains utilities for evaluating the different components
    of the harvester algorithm. 
"""
import csv 
import glob 
import os 
import signal 
import sys 

import psutil 

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

def save_dict_as_csv(dictname, filename):
    """ Converts dictionary of the format 
        {'attr1':[1, 2, 3...], 'attr2': [1, 2, ..], ...} into csv 
        with attr1, attr2, ... as headers for their resp. data. 
        Resulting csv file saved at specified location.
    """
    fields = dictname.keys()
    with open(filename, 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        for i in range(len(dictname[fields[0]])):
            row_dict = {fields[j]: dictname[fields[j]][i] for j in 
                range(len(fields))}
            writer.writerow(row_dict)