""" Configuration file for detector
    Last updated: 9/26/18
"""

darkflow_dir = '/home/jonathon/git/darkflow/' #'/root/git/darkflow/'
pb = darkflow_dir + 'built_graph/yolov2-tiny-strawb.pb'
meta = darkflow_dir + 'built_graph/yolov2-tiny-strawb.meta'
cfg = darkflow_dir + 'cfg/yolov2-tiny-strawb.cfg'
weights = darkflow_dir + 'bin/yolov2-tiny-strawb_final.weights'
labels = darkflow_dir + 'labels.txt'

# GPU config
# device = '/GPU:0'
# gpu_usage = 0.8

# CPU config
device = ''
gpu_usage = 0.0 

df_options = {
    'gpu': gpu_usage, 
    'gpuName': device,
    'labels': labels,
    'metaLoad': meta, 
    'pbLoad':pb,
    'threshold': 0.5}

tf_cfg = {
    'allow_soft_placement': True,
    'log_device_placement': False}
