# testing

Module for testing learned detector and agent. 

## Notable files
* `policy.py`: different policies to evaluate in the harvester environment
* `evaluator.py`: methods to test both the pretrained detector and DDPG policy

## How to use
Run  `evaluator.py` from the commandline.
```
cd harvester-python/testing
python evaluator.py --test 'stats'
python evaluator.py --test 'baseline' --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'
python evaluator.py --test 'canopy' --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'
python evaluator.py --test 'fixation' --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'
```
