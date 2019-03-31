# harvester-python
![harvester-gif](https://imgur.com/0dghLur.gif)

Python 2 interface for the [harvester-sim](https://github.com/jsather/harvester-sim) environment. Contains utilities for real-time strawberry detection and viewpoint optimization using deep reinforcement learning.

## Components
There are several submodules that work together for different components of the project. They work together as follows:

![dataflow](https://imgur.com/s62ti61.jpg)

* `agent`: primary interface between ROS/Gazebo and python. 

* `image`: tools for displaying and annotating the real-time camera feed.

* `detector`: used to interface with the pretrained strawberry detector using You Only Look Once, Version 2 ([YOLOv2](https://arxiv.org/abs/1612.08242)).

* `ddpg`: responsible for running Deep Deterministic Policy Gradients ([DDPG](https://arxiv.org/abs/1509.02971)) with the simulated environment. 

* `testing`: utilities and scripts for evaluating the learned agent and detector's performance.

## License
This project is licensed under the BSD 2-CLAUSE LICENSE- see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* agent interface inspired by [GPS](http://rll.berkeley.edu/gps/) and [OpenAI Gym](https://gym.openai.com/)
