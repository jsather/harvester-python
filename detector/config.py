""" Configuration file for strawberry detector. """

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

# threshold = 0.5 # needed?

# tf_cfg = {
#     'allow_soft_placement': True,
#     'log_device_placement': False}
