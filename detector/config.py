""" Configuration file for strawberry detector. """
import os

try:
    harvester_python = [pp for pp in os.environ['PYTHONPATH'].split(":")
                        if 'harvester-python' in pp][0]
except IndexError as e:
    raise IndexError(
        'Could not find harvester-python workspace.' +
        ' Did you remember to update your PYTHONPATH?')
        
network = os.path.join(harvester_python, 'detector', 'network')
pb = os.path.join(network, 'yolov2-tiny-strawb.pb')
meta = os.path.join(network, 'yolov2-tiny-strawb.meta')
cfg = os.path.join(network, 'yolov2-tiny-strawb.cfg') # needed?
labels = os.path.join(network, 'labels.txt') # needed?

# GPU config
# device = '/GPU:0'
# gpu_usage = 0.8

# CPU config
device = ''
gpu_usage = 0.0

# threshold = 0.5 # needed?

# tf_cfg = {
#     'allow_soft_placement': True,
#     'log_device_placement': False}
