""" ddpg.py is where the entire DDPG training process is run. 
    NOTE: This module is a work in progress!

    Author: Jonathon Sather
    Last updated: 9/22/2018

    Implementation inspired by:
    https://github.com/pemami4911/deep-rl/tree/master/ddpg
    https://github.com/cbfinn/gps

    TODO:
   Done?
     X  1. In replay buffer, discard trajectories that do not have enough
           samples for trace.
     X  2. Save TF model to checkpoint every "save_steps" steps
     X  3. Make configuration file so can minimize commandline args.
     X  4. Add reset condition so can refresh Gazebo every so often..
           Will this be easier if have agent on this machine? I think so, actually.
           Work on setting this up through pubsub and then can restart agent as needed!
     X  5. Update display to only show what I want to print (except pesky Moveit...)
     X  6. Set up GCE instance and install Tensorflow-gpu.
     X  7. Set up communication channels between DDPG on GCE instance and agent
           on laptop. "What do I need to communicate to and from laptop?"
     X  8. Put DDPG on GCE instance and debug general communication issues.
     X  9. Double check algorithm with paper.
     X 10. Redo look up table OR update path planning to loop around far side so
           can use to move to random locations. Then just need to save reset
           joint angles. Perhaps update lookup table later.
     X 11. Add 'headless' running option, and see if can switch between headless
           and display.
     X 12. Train DDPG agent and see if can learn reasonable policy - note how
           long it takes. **Do writeup here if promising results!**
   WIP 13. Set up on single instance to display in browser. See how fast can train.
       14. Add additional data logging metrics: # training iter, # steps per episode.
       15. Modify reward: add action magnitude penalty and use max image size(800x800).
       16. Make DDPGfD improvements: expert demos, priority replay buffer. Test here.
           Again, do writeup here if promising results!
       17. Add my novel addition of DQfD pretraining + transfer learning.
       18. Report results.
   WIP 19. Trim the bs, add config options to train on CPU or GPU, make project into package.
       20. Improve documentation, put on github/gitlab (private for now).
       21. Celebrate/get In-N-Out
"""
from __future__ import print_function
import argparse
import copy
import datetime
import glob
import os
import pickle
import random
import signal
import sys
import time

import numpy as np
import tensorflow as tf

import ddpg_agent 
import config as ddpg_cfg
import agent.config as agent_cfg
from networks import ActorNetwork, CriticNetwork, EmbeddingNetwork
from replay_buffer import ReplayBuffer
from noise import OrnsteinUhlenbeckActionNoise

import pdb

#------------------------------ Helper functions ------------------------------#
def get_latest(weights_dir):
    """ Gets latest weights file and var pickle from specified directory. """
    dir_contents = glob.glob(weights_dir + '/*')
    while dir_contents:
        latest = max(dir_contents, key=os.path.getctime)
        weights_file_list = glob.glob(latest + '/ddpg*.meta')
        vars_file = latest + '/training_vars.pkl'

        if weights_file_list and glob.glob(vars_file):
            weights_file = max(weights_file_list, key=os.path.getctime)[:-5]

            good_pickle = True
            with open(vars_file) as f:
                try:
                    pickle.load(f)
                except ValueError:
                    good_pickle = False

            if good_pickle:
                return weights_file, vars_file
        dir_contents.remove(latest)
    return '', ''

def update_cfg(args):
    """ Updates values in global configuration to match current configuration.
    """
    cfg_dict = ddpg_cfg.__dict__
    for key in sorted(cfg_dict.keys()):
        update_cfg = False
        try:
            if args[key.lower()] != cfg_dict[key]:
                update_cfg = True
        except KeyError:
            continue

        if update_cfg:
            if isinstance(args[key.lower()], basestring):
                new_val = "'" + str(args[key.lower()]) + "'"
            else:
                new_val = str(args[key.lower()])

            exec('ddpg_cfg.' + str(key) + ' = ' + new_val) # See if there is a "safer" way to do this...

def set_logfile():
    """ Sets stdout and stderr to file (if specified). """
    if ddpg_cfg.logfile != '':
        sys.stdout = open(ddpg_cfg.logfile, 'a')
        sys.stderr = open(ddpg_cfg.logfile, 'a')

#-------------------------------- Main class ----------------------------------#
class DDPG(object):
    """ Object responsible for running the DDPG algorithm. Could also
        potentially be used for testing (we'll see... may make a separate class)
    """

    def __init__(self, session, actor, critic):
        """ Initialize training and testing objects and configuration. """
        # Set up training objects
        self.session = session
        self.actor = actor
        self.critic = critic

        # Load training config
        self.batch_size = ddpg_cfg.batch_size
        self.pretrain_steps = ddpg_cfg.pretrain_steps
        self.max_episodes = ddpg_cfg.max_episodes
        self.max_ep_steps = ddpg_cfg.max_episode_len
        self.save_freq = ddpg_cfg.save_freq
        self.output_dir = ddpg_cfg.results_dir
        self.weights_file = ddpg_cfg.weights_file
        self.vars_file = ddpg_cfg.vars_file

        # Set up summary and save ops
        self.summary_dir = os.path.join(self.output_dir,
            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.save_config()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        self.ckpt_file = os.path.join(self.summary_dir, 'ddpg')
        self.summary_ops, self.summary_vars = self.build_summaries()

        # Set up graph and training variables
        self.session.run(tf.global_variables_initializer())

        if self.weights_file:
            self.restore_weights()

        if self.vars_file:
            self.restore_training_variables()
        else:
            self.replay_memory = ReplayBuffer(ddpg_cfg.buffer_size,
                ddpg_cfg.buffer_dir)
            self.current_episode = 0
            self.total_steps = 0

        self.update_target_networks()
        self.writer = tf.summary.FileWriter(self.summary_dir,
            self.session.graph)

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        # Initialize agent
        self.agent = ddpg_agent.DDPGAgent(headless=ddpg_cfg.headless)

    def build_summaries(self):
        """ Sets up summary operations for use with Tensorboard. """
        episode_reward = tf.Variable(0.)
        tf.summary.scalar('Reward', episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        tf.summary.scalar('Qmax_Value', episode_ave_max_q)
        episode_steps = tf.Variable(0.)
        tf.summary.scalar('Steps', episode_steps)

        summary_vars = [ # Added network inputs for variable summaries
            episode_reward, 
            episode_ave_max_q, 
            episode_steps, 
            self.actor.obs_in, 
            self.actor.pos_in, 
            self.actor.phase,
            self.actor.batch_size, 
            self.critic.obs_in, 
            self.critic.pos_in, 
            self.critic.action, 
            self.critic.phase,
            self.critic.batch_size]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    def save_config(self):
        """ Save training configuration. """
        with open(os.path.join(self.summary_dir, 'config.txt'), 'w') as f:
            cfg_dict = ddpg_cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

    def restore_weights(self):
        """ Restores weights from saved location, if it exists. """
        print('{} Restoring weights from: {}...'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
            self.weights_file), end='')
        self.saver.restore(self.session, self.weights_file)
        print('Done.')
        sys.stdout.flush()

    def restore_training_variables(self):
        """ Restores training variables from saved location. """
        with open(self.vars_file) as f:
            print('{} Restoring training variables from: {}...'.format(
                datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                self.vars_file), end='')
            var_dict = pickle.load(f)
            print('Done.')

        self.current_episode = var_dict['current_episode']
        self.total_steps = var_dict['total_steps']
        self.replay_memory = var_dict['replay_memory']
        self.replay_memory.update_inventory()

        print('{} Continuing at episode {} with total steps {}.'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
            self.current_episode, self.total_steps))
        sys.stdout.flush()

    def save_training_variables(self):
        """ Saves training variables in summary directory. """
        print('{} Saving training variables to: {}...'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
            self.summary_dir), end='')
        var_dict = {
            'current_episode': self.current_episode,
            'total_steps': self.total_steps,
            'replay_memory': self.replay_memory}
        save_loc = os.path.join(self.summary_dir, 'training_vars.pkl')
        with open(save_loc, 'w') as f:
            pickle.dump(var_dict, f)
        print('Done.')
        sys.stdout.flush()

    def save_session(self):
        """ Saves session at checkpoint location. """
        print('{} Saving checkpoint to: {}...'.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
            self.summary_dir), end='')
        sys.stdout.flush()
        self.saver.save(self.session, self.ckpt_file, 
            global_step=self.total_steps)
        print('Done.')
        sys.stdout.flush()

    def update_target_networks(self):
        """ Updates the target networks of the actor and critic. """
        self.actor.update_target_network()
        self.critic.update_target_network()

    def log_episode_data(self, reward, ep_ave_max_q, ep_train_iter, ep, steps):
        """ Logs episode data for Tensorboard and prints metrics to console. """

        # NOTE: Batch needed for variable summaries
        batch = self.replay_memory.sample_batch(self.batch_size)
        o_batch = batch[0]
        j_batch = batch[1]
        a_batch = batch[2]
        actual_bs = o_batch.shape[0]

        try:
            q = ep_ave_max_q / ep_train_iter
        except ZeroDivisionError:
            q = 0.
        
        summary_str = self.session.run(
            self.summary_ops,
            feed_dict={
                self.summary_vars[0]: reward,
                self.summary_vars[1]: float(q), 
                self.summary_vars[2]: float(steps), 
                self.summary_vars[3]: o_batch, 
                self.summary_vars[4]: j_batch, 
                self.summary_vars[5]: 0,
                self.summary_vars[6]: actual_bs,
                self.summary_vars[7]: o_batch, 
                self.summary_vars[8]: j_batch, 
                self.summary_vars[9]: a_batch, 
                self.summary_vars[10]: 0,
                self.summary_vars[11]: actual_bs})
        self.writer.add_summary(summary_str, ep)
        self.writer.flush()

        print('| Reward: {:4f} | Steps: {:d} | Episode {:d} | Qmax: {:4f}'.format(
            reward, steps, ep, q))
        sys.stdout.flush()
    
    def process_info(self, info):
        """ Processes info dictionary from agent and takes appropriate 
            action.
        """
        if info['valid'] and info['fraction'] == 0.0:
            raise Exception('Invalid agent state.') 

    def train_networks(self, update_steps=1):
        """ Actor and critic networks by sampling data from replay memoryself.
            Returns average max q value for each training step.
        """
        sum_max_q = 0.

        print('{} Training networks... '.format(
            datetime.datetime.now().strftime('%m-%d %H:%M:%S')), end='')
        sys.stdout.flush()

        start = time.time()

        for update in range(update_steps):
            batch = self.replay_memory.sample_batch(
                self.batch_size)
            o_batch = batch[0]
            j_batch = batch[1]
            a_batch = batch[2]
            r_batch = batch[3]
            t_batch = batch[4]
            o2_batch = batch[5]
            j2_batch = batch[6]

            # Get actual batch size in case partial batch
            actual_bs = o_batch.shape[0]

            # Calculate target values
            target_q = self.critic.predict_target(
                o2_batch,
                j2_batch,
                self.actor.predict_target(
                    o2_batch,
                    j2_batch,
                    batch_size=actual_bs),
                batch_size=actual_bs)

            # Calculate training values
            end_mask = -(t_batch - 1)
            y_i = r_batch + self.critic.gamma*target_q*end_mask

            # Update critic given targets a la Q-learning
            predicted_q_value, _ = self.critic.train(
                o_batch,
                j_batch,
                a_batch,
                y_i,
                batch_size=actual_bs)
            sum_max_q += np.amax(predicted_q_value)

            # Update actor policy using DPG
            a_outs = self.actor.predict(
                o_batch,
                j_batch,
                batch_size=actual_bs,
                add_noise=False)
            grads = self.critic.action_gradients(
                o_batch,
                j_batch,
                a_outs,
                batch_size=actual_bs)

            self.actor.train(
                o_batch,
                j_batch,
                grads[0],
                batch_size=actual_bs)

            self.update_target_networks()

        elapsed = time.time() - start

        print('Done. (Elapsed: ' +  str(elapsed) + 's)')
        sys.stdout.flush()

        ave_max_q = sum_max_q / update_steps
        return ave_max_q
      
    def reset(self, timeout=30):
        """ Reset in environment.
            Args:
                timeout: int seconds, defaults to 30s to account for plant
                         spawning 
        """
        self.agent.reset() 

        ret = None 
        start = time.time() 

        while ret is None: 
            if time.time() - start > timeout: 
                raise Exception('Timeout error: No response from agent.')
            ret = self.agent.get_state() 
            time.sleep(0.1) 
        
        return ret 
            
    def reset_and_train(self, timeout=30):
        """ Reset in environment and trains in parallel. 
            Args:
                timeout: int seconds, defaults to 30 to account for plant
                         spawning 
        """
        self.agent.reset() 

        sum_ave_max_q = 0. 
        train_iter = 0 
        ret = None 
        start = time.time() 

        print('DEBUG: Reset and train in parallel')
        sys.stdout.flush() 

        while ret is None: 
            if time.time() - start > timeout: 
                raise Exception('Timeout error: No response from agent.')
            sum_ave_max_q += self.train_networks(update_steps=1) 
            train_iter += 1 
            print('Train iter: ' + str(train_iter))

            ret = self.agent.get_state() 
        
        return ret, sum_ave_max_q, train_iter 

    def step(self, action, timeout=10):
        """ Takes step in environment. """ 
        self.agent.step(action) 

        ret = None 
        start = time.time() 

        while ret is None: 
            if time.time() - start > timeout: 
                raise Exception('Timeout error: No response from agent.')
            ret = self.agent.get_return() 
            time.sleep(0.1)
        
        return ret 

    def step_and_train(self, action, timeout=10):
        """ Takes step in environment and trains in parallel.  """
        self.agent.step(action)
        
        sum_ave_max_q = 0. 
        train_iter = 0 
        ret = None 
        start = time.time() 

        print('DEBUG: Step and train in parallel')
        sys.stdout.flush()

        while ret is None: 
            if time.time() - start > timeout: 
                raise Exception('Timeout error: No response from agent.')
            sum_ave_max_q += self.train_networks(update_steps=1)
            train_iter += 1 
            print('Train iter: ' + str(train_iter))

            ret = self.agent.get_return() 
        
        return ret, sum_ave_max_q, train_iter 

    def learn(self):
        """ Runs the DDPG algorithm. """
        start = copy.copy(self.current_episode)

        for ep in range(start, self.max_episodes):
            if ep % self.save_freq == 0:
                self.save_session()
                self.save_training_variables()

            print('{} Starting episode {}'.format(
                datetime.datetime.now().strftime('%m-%d %H:%M:%S'), ep))
            sys.stdout.flush()

            self.current_episode = copy.copy(ep)
            ep_reward = 0.
            ep_ave_max_q = 0.
            ep_train_iter =0

            if self.total_steps < self.pretrain_steps:
                o, j = self.reset()
            else:
                [o, j], q, iter = self.reset_and_train()
                ep_train_iter += iter
                ep_ave_max_q += q
            
            self.actor.OU_noise.reset() # reset noise every episode (new)

            for step in range(self.max_ep_steps):
                o_reshape = np.reshape(o, (1, self.actor.obs_shape[0],
                    self.actor.obs_shape[1],
                    self.actor.obs_shape[2]))
                j_reshape = np.reshape(j, (1, self.actor.pos_shape[0]))

                # Run actor network forward
                a = self.actor.predict(o_reshape, j_reshape, batch_size=1,
                    training=0)
                
                # Sample from buffer and train between each step #TODO: Consider ignoring step if too short (info['elapsed'] < val)
                if self.total_steps < self.pretrain_steps:
                    [o2, j2], r, t, info = self.step(a)
                else:
                    [[o2, j2], r, t, info], q, iter = self.step_and_train(a)
                    ep_train_iter += iter
                    ep_ave_max_q += q
                
                # self.process_info(info)

                # Add to replay buffer
                a = np.reshape(a, -1)
                self.replay_memory.add(o, j, a, r, t, o2, j2)
                self.total_steps += 1

                o = o2
                j = j2
                ep_reward += r

                if t or step == (self.max_ep_steps - 1):
                    print('{} Episode {} finished.'.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        ep))
                    print('{} Total steps: {}'.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.total_steps))

                    # Log data and start new episode
                    ep_steps = step + 1
                    self.log_episode_data(
                        reward=ep_reward,
                        ep_ave_max_q=ep_ave_max_q,
                        ep_train_iter=ep_train_iter,
                        ep=ep,
                        steps=ep_steps)
                    break

    def exit_gracefully(self, sig, frame):
        """ Save configuration before exit. """
        print('Signal: ' + str(sig))
        self.save_session()
        self.save_training_variables()
        self.session.close()
        sys.exit()

def main():
    # Parse command-line arguments and run algorithm
    parser = argparse.ArgumentParser(
        description='provide arguments for DDPG algorithm')

    # Learning parameters
    parser.add_argument('--buffer-size',
        help='max size of the replay buffer',
        default=ddpg_cfg.buffer_size,
        type=int)
    parser.add_argument('--batch-size',
        help='size of minibatch for minibatch-SGD',
        default=ddpg_cfg.batch_size,
        type=int)
    parser.add_argument('--pretrain-steps',
        help='number of steps in environment before start training prodecure',
        default=ddpg_cfg.pretrain_steps,
        type=int)

    # Experiment parameters
    parser.add_argument('--np-seed',
        help='numpy random seed',
        default=ddpg_cfg.np_seed,
        type=int)
    parser.add_argument('--tf-seed',
        help='tensorflow random seed',
        default=ddpg_cfg.tf_seed,
        type=int)
    parser.add_argument('--max-episodes',
        help='max num of episodes to do while training',
        default=ddpg_cfg.max_episodes,
        type=int)
    parser.add_argument('--max-episode-len',
        help='max length of 1 episode',
        default=ddpg_cfg.max_episode_len,
        type=int)
    parser.add_argument('--save-freq',
        help='number of episodes between saving each checkpoint',
        default=ddpg_cfg.save_freq,
        type=int)
    parser.add_argument('--results-dir',
        help='directory for logging training info',
        default=ddpg_cfg.results_dir,
        type=str)
    parser.add_argument('--buffer-dir',
        help='directory containing replay buffer',
        default=ddpg_cfg.buffer_dir,
        type=str)
    parser.add_argument('--weights-file',
        help='file containing pretrained weights (optional)',
        default=ddpg_cfg.weights_file,
        type=str)
    parser.add_argument('--vars-file',
        help='pkl file containing existing dynamic variables (optional)',
        default=ddpg_cfg.vars_file,
        type=str)
    parser.add_argument('--continue-training',
        help='continue training with latest weights file',
        default=False,
        action='store_true')
    parser.add_argument('--headless', 
        help='run simulation and camera feed headless', 
        default=ddpg_cfg.headless, 
        action='store_true')
    parser.add_argument('--logfile',
        help='file to display commandline output (optional)',
        default=ddpg_cfg.logfile,
        type=str)
    args_dict = vars(parser.parse_args())

    if args_dict['continue_training']:
        args_dict['weights_file'], args_dict['vars_file'] = \
            get_latest(args_dict['results_dir'])
        if not args_dict['weights_file']:
            print('No recent weights file and/or vars file found. ', end='')
            print('Starting from scratch.')

    update_cfg(args_dict)
    set_logfile()
    config = tf.ConfigProto(**ddpg_cfg.tf_cfg)
    config.gpu_options.allow_growth = True
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
        try:
            ddpg.learn()
        except Exception as e:
            ddpg.exit_gracefully(e, None)

if __name__ == '__main__':
    main()
else: 
    main() # lol