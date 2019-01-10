""" Configuration file for detector
    Last updated: 9/26/18
"""

device = '/GPU:0'

darkflow_dir = '/root/git/darkflow/'
pb = darkflow_dir + 'built_graph/yolov2-tiny-strawb.pb'
meta = darkflow_dir + 'built_graph/yolov2-tiny-strawb.meta'
cfg = darkflow_dir + 'cfg/yolov2-tiny-strawb.cfg'
weights = darkflow_dir + 'bin/yolov2-tiny-strawb_final.weights'
labels = darkflow_dir + 'labels.txt'

df_options = {
    'gpu': 0.8, 
    'gpuName': device,
    'labels': labels,
    'metaLoad': meta, 
    'pbLoad':pb,
    'threshold': 0.5}

tf_cfg = {
    'allow_soft_placement': True,
    'log_device_placement': False}

