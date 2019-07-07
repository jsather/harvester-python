""" Configuration file for strawberry detector. """
import imp
import os

project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
global_config = imp.load_source('config',
    os.path.join(project_dir, 'config.py'))

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

if global_config.gpu:
    device = '/GPU:0'
    gpu_usage = 0.8
else:
    device = ''
    gpu_usage = 0.0
