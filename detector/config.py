""" Configuration file for strawberry detector. """

# darkflow_dir = '/home/jonathon/git/darkflow/' #'/root/git/darkflow/'
# pb = darkflow_dir + 'built_graph/yolov2-tiny-strawb.pb'
# meta = darkflow_dir + 'built_graph/yolov2-tiny-strawb.meta'
# cfg = darkflow_dir + 'cfg/yolov2-tiny-strawb.cfg'
# weights = darkflow_dir + 'bin/yolov2-tiny-strawb_final.weights'
module_dir = '/home/jonathon/_git/harvester-python/detector/'
pb = module_dir + 'network/yolov2-tiny-strawb.pb'
meta = module_dir + 'network/yolov2-tiny-strawb.meta'
cfg = module_dir + 'network/yolov2-tiny-strawb.cfg' # needed?
labels = module_dir + 'network/labels.txt' # needed?

# GPU config
# device = '/GPU:0'
# gpu_usage = 0.8

# CPU config
device = ''
gpu_usage = 0.0 

threshold = 0.5 # needed?

# df_options = {
#     'gpu': gpu_usage, 
#     'gpuName': device,
#     'labels': labels,
#     'metaLoad': meta, 
#     'pbLoad':pb,
#     'threshold': 0.5}

tf_cfg = {
    'allow_soft_placement': True,
    'log_device_placement': False}
