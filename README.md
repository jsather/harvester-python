# harvester-python
![harvester-gif](https://imgur.com/0dghLur.gif)

Python 2 interface for the [harvester-sim](https://github.com/jsather/harvester-sim) environment. Contains utilities for real-time strawberry detection and viewpoint optimization with deep reinforcement learning. Originally used by Jonathon Sather in his research on autonomous strawberry harvesting ([paper](https://arxiv.org/abs/1903.02074)) ([video summary](https://youtu.be/C6hrCVv2B-o)).


## Dependencies
* [harvester-sim](https://github.com/jsather/harvester-sim)
* [tensorflow](https://www.tensorflow.org/)
* [open-cv](https://pypi.org/project/opencv-python/)

## Installation
Clone the repo, and add it to your python path.
```
cd ~/git
git clone https://github.com/jsather/harvester-python.git
export PYTHONPATH=~/git/harvester-python:$PYTHONPATH
```

Then run `setup.py` to build the detector cython modules (adapted from [Darkflow](https://github.com/thtrieu/darkflow)).
```
cd ~/git/harvester-python
python setup.py build_ext --inplace
```

## Components
There are five submodules that work together for different components of the project. They work together as follows:

<img src="https://imgur.com/s62ti61.jpg" width="600">


* `agent`: primary interface between ROS/Gazebo and python. 

* `image`: tools for displaying and annotating the real-time camera feed.

* `detector`: used to interface with the pretrained strawberry detector using You Only Look Once, Version 2 ([YOLOv2](https://arxiv.org/abs/1612.08242)).

* `ddpg`: responsible for running Deep Deterministic Policy Gradients ([DDPG](https://arxiv.org/abs/1509.02971)) with the simulated environment. 

* `testing`: utilities and scripts for evaluating the learned agent and detector's performance.

## Usage
### Training
Train the agent by running `train.py`:
```
cd ~/git/harvester-python
python train.py
```

You should see a window titled "harvester vision" showing the agent's current viewpoint:

<img src="https://imgur.com/8AkRTUm.jpg" width="450">

### Other
For running individual components or testing, refer to the documentation for the relevant submodule. 

## License
This project is licensed under the BSD 2-CLAUSE LICENSE- see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* agent interface inspired by [GPS](http://rll.berkeley.edu/gps/) and [OpenAI Gym](https://gym.openai.com/)
* detector post processing adapted from [Darkflow](https://github.com/thtrieu/darkflow)  
