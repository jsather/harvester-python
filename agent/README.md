# agent
## Notable files
`agent_ros.py`: agent interfaces to communicate with the simulated harvester robot. Contains methods like "step" and "reset", which use an API similar to [OpenAI Gym](https://gym.openai.com/), as well as to other class-specific methods.
`plant_ros.py`: interface to spawn and destroy plants in the simulated environment.
`make_hemi_lut.py`: utility for creating look-up-table for joint angles corresponding to positions on hemisphere above starwberry plant.
`start_agent.py`: script for running agent_ros node. Typically executed in the background, so can run asynchronously with reinforcement learning training procedure.

## Adding own agent
To add your own agent subclass, follow the template of `agent_ros.AgentROS` and make your own methods where indicated. 

## Example usage
Initialize environment and take step:
```
import agent_ros.HemiAgentROS as agent

agent1 = agent()
agent1.reset()
ret = agent1.step([0.1, 0.1])
print(ret)
```
