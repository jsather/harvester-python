""" replay_buffer.py contains the class ReplayBuffer, which can be used to
    create an experience replay for Q-learning.

    Author: Jonathon Sather
    Last updated: 9/21/2018

    Implementation inspired by:
    https://github.com/pemami4911/deep-rl/tree/master/ddpg

    New: Used replay buffer that points to file location. NOTE: Episode replay
         buffer has not yet been updated to do this!
"""

import collections
import copy
import datetime
import glob
import os
import pickle
import random

import numpy as np

class EpisodeReplayBuffer(object):
    """ Replay buffer object used for training recurrent neural network. Stores
        'buffer_size' trajectories with methods to add and remove elements at
        specified trace lengths.
    """

    def __init__(self, buffer_size, init_data=[]):
        """ Initialize experience replay buffer. """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = collections.deque(init_data)

    def add(self, episode):
        """ Adds a trajectory from one episode to the buffer. Removes oldest
            trajectory if buffer is full.
        """
        if self.count < self.buffer_size:
            self.buffer.append(episode)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(episode)

    def size(self):
        """ Returns number of elements (episodes) stored in replay buffer. """
        return self.count

    def sample_batch(self, batch_size, trace_length):
        """ Samples from 'batch_size' trajectories to create an array of traces
            of length 'trace_length'.
        """
        batch = []

        if self.count < batch_size:
            batch_size = self.count

        episodes = random.sample(self.buffer, batch_size)

        for ep in episodes:
            try:
                start = np.random.randint(0, len(ep)+1-trace_length)
            except ValueError:
                batch_size -= 1
                print(
                    "Episode length: " + str(len(ep)) +
                    " too short for trace length: " + str(trace_length) +
                    "=> Omitting from batch. New batch size: " +
                    str(batch_size))
            else:
                batch.append(ep[start:start+trace_length])

        batch = np.reshape(np.array(batch), [batch_size*trace_length, -1])

        o_batch = np.stack(batch[:,0], axis=0)
        j_batch = np.vstack(batch[:,1])
        a_batch = np.vstack(batch[:,2])
        t_batch = np.vstack(batch[:,3])
        r_batch = np.vstack(batch[:,4])
        o2_batch = np.stack(batch[:,5], axis=0)
        j2_batch = np.vstack(batch[:,6])

        return (o_batch, j_batch, a_batch, t_batch, r_batch, o2_batch, j2_batch)

    def clear(self):
        """ Clears replay buffer and resets count. """
        self.buffer.clear()
        self.count = 0

class ReplayBuffer(object):
    """ Replay buffer object used for training neural network. Stores
       'buffer_size' elements in buffer with methods to add and remove elements.
    """

    def __init__(self, buffer_size, folder):
        """ Initialize experience replay buffer. """
        self.buffer_size = buffer_size
        self.folder = folder

        self.update_inventory()
        self._name_idx = 0 # Appended to new filenames to ensure uniqueness

    def update_inventory(self):
        """ Updates memberdata to match contents in buffer folder. """

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        files = glob.glob(os.path.join(self.folder, '*.pkl'))
        files.sort(key=os.path.getmtime)
        try:
            self.buffer = collections.deque(files)
        except TypeError:
            self.buffer = collections.deque()

        self.count = len(self.buffer)
        self.trim_buffer()

    def generate_filename(self):
        """ Uses current time and name index to generate unique filename for
            new element.
        """
        date = datetime.datetime.now().strftime('%m-%d_%H:%M:%S')
        relname = date + '_' + str(self._name_idx) + '.pkl'
        self._name_idx = (self._name_idx + 1) % int(1e10) # Arbitrary cutoff so doesn't grow indefinitely

        return os.path.join(self.folder, relname)

    def trim_buffer(self):
        """ Removes excess elements from buffer, if they exist. """
        for i in range(self.count - self.buffer_size):
            os.remove(self.buffer.popleft())

        self.count = len(self.buffer)

    def pickle_experience(self, experience, file):
        """ Pickles experience at specified location. """
        with open(file, 'w') as f:
            pickle.dump(experience, f)

    def unpickle_experience(self, file):
        """ Loads and unpickles experience from specified location. """
        with open(file) as f:
            return pickle.load(f)

    def add(self, o, j, a, r, t, o2, j2):
        """ Adds one state-transition tuple to the buffer. Removes oldest tuple
            if buffer is full.
        """
        experience = (o, j, a, r, t, o2, j2)
        file = self.generate_filename()

        self.buffer.append(file)
        self.pickle_experience(experience, file)
        self.count += 1
        self.trim_buffer()

    def size(self):
        """ Returns number of elements in replay buffer. """
        return self.count

    def sample_batch(self, batch_size):
        """ Samples 'batch_size' number of state-transition tuples from replay
            buffer. If the replay buffer has less than 'batch_size' elements,
            return all elements.
        """
        if self.count < batch_size:
            batch_files = random.sample(self.buffer, self.count)
        else:
            batch_files = random.sample(self.buffer, batch_size)

        batch = []
        for f in batch_files:
            try:
                batch.append(self.unpickle_experience(f))
            except Exception as e: # known errors: EOFError
                print(e)
                self.buffer.remove(f)

        o_batch = np.array([_[0] for _ in batch])
        j_batch = np.array([_[1] for _ in batch])
        a_batch = np.array([_[2] for _ in batch])
        r_batch = np.reshape(np.array([_[3] for _ in batch]), (-1, 1))
        t_batch = np.reshape(np.array([_[4] for _ in batch]), (-1, 1))
        o2_batch = np.array([_[5] for _ in batch])
        j2_batch = np.array([_[6] for _ in batch])

        return o_batch, j_batch, a_batch, r_batch, t_batch, o2_batch, j2_batch

    def clear(self):
        """ Clears replay buffer and resets count. """
        _buffer_size = copy.copy(self.buffer_size)
        self.buffer_size = 0
        self.trim_buffer()
        self.buffer_size = copy.copy(_buffer_size)
