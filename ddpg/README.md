# ddpg 

Implementation of Deep Deterministic Policy Gradients ([DDPG](https://arxiv.org/abs/1509.02971)) to interface with the simulated agent. 

## Notable files
* `ddpg.py`: where the main algorithm lives
* `networks.py`: actor and critic networks classes
* `replay_buffer.py`: experience replay classes
* `noise.py`: Ornstein-Uhlenbeck noise class
* `ddpg_agent.py`: ROS communication node to start and communicate with agent

## How to train
Ideally, start/continue training through the WatchDog class (one folder level up). This approach periodically checks to make sure all components of the system are running, and restarts/continues the training process as needed.

```
cd harvester-python
python train.py --restart-training --headless
```

Can also run training directly by creating DDPG object and running `learn` method. Example:

```
   # Make sure start within tf session
   with tf.Session(config=config) as session:
        np.random.seed(ddpg_cfg.np_seed)
        tf.set_random_seed(ddpg_cfg.tf_seed)

        # Agent info
        [obs_shape, action_shape] = agent_cfg.hemi_state_shape
        action_bound = agent_cfg.hemi_action_bound
        OU_noise = OrnsteinUhlenbeckActionNoise(
            mu=agent_cfg.mu,
            sigma=agent_cfg.sigma,
            theta=agent_cfg.theta)

        # Initialize function approximators
        # embedding_network = EmbeddingNetwork(session)
        embedding_network = None
        actor_network = ActorNetwork(
            session,
            obs_shape,
            action_shape,
            action_bound,
            OU_noise,
            embedding=embedding_network)
        critic_network = CriticNetwork(
            session,
            obs_shape,
            action_shape,
            embedding=embedding_network)

        ddpg = DDPG(session, actor_network, critic_network)
        ddpg.learn()
```

## How to test
To test a learned policy, run the appropriate test from `evaluator.py` in harvester-python/testing. Example commands:
```
cd harvester-python/testing
python evaluator.py --test 'stats'
python evaluator.py --test 'baseline' --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'
python evaluator.py --test 'canopy' --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'
python evaluator.py --test 'fixation' --weights-file '/mnt/storage/testing/2018_10_14_16_58/ddpg-241738'
```
