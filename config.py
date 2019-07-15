""" Global config file for harvester-python. """

# Useful directories
try:
    catkin_ws = [ws for ws in os.environ['ROS_PACKAGE_PATH'].split(":")
                 if 'src' in ws][0]
except IndexError as e:
    raise IndexError(
        'Could not find catkin_ws. Have you sourced your environment?')

try:
    harvester_python = [pp for pp in os.environ['PYTHONPATH'].split(":")
                        if 'harvester-python' in pp][0]
except IndexError as e:
    raise IndexError(
        'Could not find harvester-python workspace.' +
        ' Did you remember to update your PYTHONPATH?')

storage_dir = os.path.join('harvester_python', 'storage')
gpu = False
