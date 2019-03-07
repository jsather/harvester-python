# harvester-python
![harvester-gif](https://imgur.com/0dghLur.gif)

Python interface for the harvester-sim environment. Contains utilities for real-time strawberry detection and dynamic detection using reinforcement learning. 

## Components
There are several submodules that work together for different components of the project. 

### Agent

### Image

### Detector

### DDPG

### Testing

### Harvester-ros

Here outline the different components of the repo - agent, image, detector, ddpg, testing and harvester-ros.

go thru steps - set up firewalls 
set up disk storage
go to docker_setup section readme

docker_setup
install on instance
get latest cuda files
describe additional hacks that added
run preinstall steps - then set up nfs
run build
then deploy - show what should look like in browser

then go thru different sections of project - stating what they are used for and referring to resp readme

agent
how to test environment - go to agent class NOTE to future jon: make harvester-ros a separate repo! or not... nahh let's keep it all one
plant demo
moving agent
adding own agent subclass
refer to harvester-ros
mention inspired by aigym - link to them

detector
cite yolo, give summary + where this fits in to my algo
how to pretrain detector - add readme in detector section with explicit instructions
how to test detector - add in detector readme 

ddpg
cite ddpg, give summary, how frame problem
how to train ddpg - put in ddpg readme
how to test ddpg - put in ddpg readme

image 
how to run image annotation

harvester-ros
state that used with agent class to run simulated environment
go thru what each of the folders do 

## License
This project is licensed under the BSD 2-CLAUSE LICENSE- see the [LICENSE.md](LICENSE.md) file for details

mention darknet yolo ddpg, etc

## Acknowledgments
* agent interface inspired by [GPS](http://rll.berkeley.edu/gps/) and [OpenAI Gym](https://gym.openai.com/)
