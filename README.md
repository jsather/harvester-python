# harvester-python
![harvester-gif](https://imgur.com/0dghLur.gif)

Python interface for the [harvester-sim](https://github.com/jsather/harvester-sim) environment. Contains utilities for real-time strawberry detection and viewpoint optimization using deep reinforcement learning.

## Components
There are several submodules that work together for different components of the project. They work together as follows:

![dataflow](https://imgur.com/s62ti61.jpg)

### Agent

The agent submodule is the primary interface between ROS/Gazebo and python. It contains `agent.py` which contains classes for directly controlling the virtual harvester and interacting with the environment. 

### Image

The image submodule contains tools for displaying and annotating the real-time camera feed.

### Detector

The detector submodule is used to interface with the pretrained strawberry detector using You Only Look Once, Version 2 ([YOLOv2](https://arxiv.org/abs/1612.08242)).

### DDPG

The DDPG submodule is responsible for running Deep Deterministic Policy Gradients ([DDPG](https://arxiv.org/abs/1509.02971)) with the simulated environment. 

### Testing

This submodule contains various utilities and scripts for evaluating the learned agent and detector's performance.

## License
This project is licensed under the BSD 2-CLAUSE LICENSE- see the [LICENSE.md](LICENSE.md) file for details

mention darknet yolo ddpg, etc

## Acknowledgments
* agent interface inspired by [GPS](http://rll.berkeley.edu/gps/) and [OpenAI Gym](https://gym.openai.com/)
