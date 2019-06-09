""" networks.py contains the classes for defining the actor and critic networks
    used in the DDPG algorithm. This version does not contain recurrence units.

    Uses most parameters from original DDPG paper:
    'Continuous Control in Deep Reinforcement Learning' by Lillicrap, et al.
    (arXiv:1509.02971)

    Modular structure influence by:
    https://github.com/pemami4911/deep-rl/tree/master/ddpg
"""

import numpy as np
import tensorflow as tf

import detector.config as detect_cfg
import config as ddpg_cfg

#------------------------------ Helper functions ------------------------------#
def sqrt_unif_init(n):
    """ Creates bounded uniform weight initializer with range +-1/sqrt(n). """
    return tf.random_uniform_initializer(-1/np.sqrt(n), 1/np.sqrt(n))

def variable_summaries(var):
    """ From https://jhui.github.io/2017/03/12/TensorBoard-visualize-your-learning/ """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def dense_relu(x, size, fan_in, phase, l2_scale=0.0, scope='layer',
    batch_norm=True, add_summary=False):
    """ Creates fully connected layer with uniform initialization. Includes
        batch normalization by default.
    """
    with tf.variable_scope(scope):
        h = tf.layers.dense(x, size, activation=None,
            kernel_initializer=sqrt_unif_init(fan_in),
            bias_initializer=sqrt_unif_init(fan_in),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale),
            name='dense')
        if batch_norm:
            h = tf.contrib.layers.batch_norm(h, scale=True, is_training=phase,
                updates_collections=None, scope='bn')
        if add_summary:
            variable_summaries(h)
        return tf.nn.relu(h)

def conv_layer(x, phase, filters, stride, kernel_size, l2_scale=0.0,
    batch_norm=True, add_summary=False, scope='conv'):
    """ Creates convolutional layer with specified parameters. """
    with tf.variable_scope(scope):
        input_channels = x.get_shape().as_list()[3] # nhwc
        fan_in = kernel_size*kernel_size*input_channels*filters

        conv = tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=[kernel_size, kernel_size],
            strides=[stride, stride],
            padding='same',
            activation=None,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale),
            kernel_initializer=sqrt_unif_init(fan_in),
            bias_initializer=sqrt_unif_init(fan_in),
            name='conv')

        if batch_norm:
            conv = tf.contrib.layers.batch_norm(
                conv,
                scale=True,
                is_training=phase,
                updates_collections=None,
                scope='bn')
        
        if add_summary:
            variable_summaries(conv)
        return tf.nn.relu(conv)

def input_layer(size, phase, scope='input', batch_norm=True, 
    add_summary=False):
    """ Creates input placeholder and returns input to first hidden layer.
        Includes batch normalization by default.
    """
    with tf.variable_scope(scope):
        inputs = tf.placeholder(shape=[None, size], dtype=tf.float32,
            name='input')
        if batch_norm:
            to_h = tf.contrib.layers.batch_norm(inputs, scale=True,
                is_training=phase, updates_collections=None, scope='bn')
        else:
            to_h = inputs
        
        if add_summary:
            variable_summaries(to_h)
        return inputs, to_h

def input_embedding(embedding_net, pos_size, batch_size, phase, batch_norm=True,
    scope='input_emb'):
    """ Uses embedding network object given as input to create embedding 'layer'
        (really multiple layers). Creates input placeholders and returns input
        to first hidden fc layer.
    """
    with tf.variable_scope(scope):
        pos_in = tf.placeholder(shape=[None, pos_size], dtype=tf.float32,
            name='pos_in')
        obs_in = embedding_net.input

        embed_out = embedding_net.output
        out_num = embed_out.shape[1]*embed_out.shape[2]*embed_out.shape[3]
        embed_reshape = tf.reshape(embed_out, [batch_size, out_num])
        to_fc = tf.concat([embed_reshape, pos_reshape], 1) # NOTE to future Jon: where did pos_reshape go?

        if batch_norm:
            to_fc = tf.contrib.layers.batch_norm(to_fc, scale=True,
                is_training=phase, updates_collections=None, scope='bn')

        return obs_in, pos_in, to_fc

def input_conv(obs_shape, pos_size, filters, strides, kernel_sizes, batch_size,
    phase, add_summary=False, scope='input_conv'):
    """ Creates series of conv layers using default parameters. Creates input
        placeholders and returns input to first hidden fc layer. 'filters'
        input is a list of filter sizes for each layer. The number of layers is
        inferred from len(filters).
    """
    with tf.variable_scope(scope):
        pos_in = tf.placeholder(shape=[None, pos_size], dtype=tf.float32,
            name='pos_in')
        obs_in = tf.placeholder(
            shape=[None, obs_shape[0], obs_shape[1], obs_shape[2]],
            dtype=tf.float32, name='obs_in')

        conv = obs_in
        for layer in range(len(filters)):
            conv = conv_layer(
                conv,
                phase,
                filters[layer],
                strides[layer],
                kernel_sizes[layer],
                scope='conv'+str(layer), 
                add_summary=add_summary)

        _, m, n, c = conv.get_shape().as_list()
        conv_reshape = tf.reshape(conv, [batch_size, m*n*c])
        to_fc = tf.concat([conv_reshape, pos_in], 1)

        return obs_in, pos_in, to_fc

def output_layer(x, size, unif_mag, act=None, add_summary=False, scope='output'):
    """ Creates output layer with tanh nonlinearity and uniform initialization.
    """
    with tf.variable_scope(scope):
        out = tf.layers.dense(x, size, activation=act,
            kernel_initializer=tf.random_uniform_initializer(
                -unif_mag, unif_mag),
            bias_initializer=tf.random_uniform_initializer(-unif_mag, unif_mag),
            name='out')
        if add_summary:
            variable_summaries(out) # Add when restart training
        return out

#------------------------------ Network classes -------------------------------#
class EmbeddingNetwork(object):
    """ Defines the embedding network which takes in a camera image and outputs
        a low-dimensional embedding which can be used as input to the actor
        or critic networks.
        NOTE: Not implemented yet!
    """

    def __init__(self, session=None):
        """ Initialize the embedding network using Darkflow to port the Darknet
            model to Tensorflow.
        """
        raise NotImplementedError('Embedding class not implemented') 
        # self.session = session
        
        # with tf.device(ddpg_cfg.device):
        #     self.net = build.TFNet(detect_cfg.df_options) 
        #     self.input = self.net.inp 
        #     self.output = self.net.out 
        #     self.params = tf.trainable_variables()

    def preprocess(self, input):
        """ Preprocesses input image(s) to be ready for embedding. """
        raise NotImplementedError('Embedding class not implemented') 
        # self.net.framework.preprocess(input)

    def create_feed(self, inputs):
        """ Preprocesses input image(s) to be ready for embedding. Expects a
            list of inputs.
        """
        raise NotImplementedError('Embedding class not implemented') 
        # feed = map(lambda x: np.expand_dims(self.preprocess(x), 0), inputs)
        # return np.concatenate(feed, 0)

    def embed(self, input):
        """ Embed image by running through network. This method is primarily
            used for debugging purposes, as this operation will likely be
            integrated into the 'run' methods of the actor and critic.
            args:
                input: List of cv2 images
            returns:
                embedded: Numpy array of embedded images
        """
        raise NotImplementedError('Embedding class not implemented') 
        # processed = self.create_feed(input)
        # embedded = self.session.run(self.output_flattened,
        #     feed_dict={self.input: processed})
        # return embedded

    def get_num_variables(self):
        """ Returns number of parameters associated with embedding network.
        """
        raise NotImplementedError('Embedding class not implemented')  
        # return len(self.params)

class ActorNetwork(object):
    """ Defines the actor network (Q-network optimizer) used in the DDPG
        algorithm.
    """

    def __init__(self, session, obs_shape, action_shape, action_bound,
        noise, embedding=None):
        """ Initialize actor and target networks and update methods. """
        self.session = session
        self.obs_shape = obs_shape
        self.pos_shape = action_shape # position component of state
        self.action_shape = action_shape
        self.action_bound = action_bound
        self.OU_noise = noise # noise process for exploration
        self.embedding = embedding # embedding network (or None)

        self.learning_rate = ddpg_cfg.actor_lr
        self.tau = ddpg_cfg.tau

        if self.embedding is not None:
            self.hidden_1_size = ddpg_cfg.hidden_1_size_embedding
            self.hidden_2_size = ddpg_cfg.hidden_2_size_embedding
            self.out_init_mag =  ddpg_cfg.out_init_mag_embedding
        else:
            self.filters_per_layer = ddpg_cfg.filters_per_layer # list
            self.stride_per_layer = ddpg_cfg.stride_per_layer # list
            self.kernel_size_per_layer = ddpg_cfg.kernel_size_per_layer # list

            self.hidden_1_size = ddpg_cfg.hidden_1_size_conv
            self.hidden_2_size = ddpg_cfg.hidden_2_size_conv
            self.out_init_mag = ddpg_cfg.out_init_mag_conv
        
        with tf.device(ddpg_cfg.device):
            self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_act')
            self.phase = tf.placeholder(tf.bool, name='phase_act')

            # Initialize actor network
            num_vars = len(tf.trainable_variables())
            self.obs_in, self.pos_in, self.out, self.scaled_out = \
                self.create_actor_network(add_summaries=True)
            self.network_params = tf.trainable_variables()[num_vars:]

            # Initialize target actor network
            num_vars = len(tf.trainable_variables())
            self.target_obs_in, self.target_pos_in, self.target_out, \
                self.target_scaled_out = self.create_actor_network(prefix='tar_')
            self.target_network_params = tf.trainable_variables()[num_vars:]

            # Define target update op
            self.update_target_network_params = \
                [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(self.target_network_params[i], 1.0 - self.tau))
                for i in range(len(self.target_network_params))]

            # Define ops for getting necessary gradients
            self.action_gradient = \
                tf.placeholder(tf.float32, [None, self.action_shape[0]])
            self.unnormalized_actor_gradients = tf.gradients(
                self.scaled_out, self.network_params, -self.action_gradient)
            self.actor_gradients = list(map(lambda x: tf.div(x,
                tf.cast(self.batch_size, tf.float32)),
                self.unnormalized_actor_gradients))

            # Define optimization op
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self, prefix='', add_summaries=False):
        """ Constructs the actor network. Phase is boolean tensor corresponding
            to whether in training or testing phase.
        """
        if self.embedding is not None:
            obs_in, pos_in, to_fc = input_embedding(
                self.embedding,
                self.pos_shape[0],
                self.batch_size,
                self.phase,
                scope=prefix+'input_emb_act')
        else:
            obs_in, pos_in, to_fc = input_conv(
                self.obs_shape,
                self.pos_shape[0],
                self.filters_per_layer,
                self.stride_per_layer,
                self.kernel_size_per_layer,
                self.batch_size,
                self.phase,
                add_summary=add_summaries,
                scope=prefix+'input_conv_act')

        fc1 = dense_relu(to_fc, self.hidden_1_size,
            to_fc.get_shape().as_list()[1], self.phase, add_summary=add_summaries,
            scope=prefix+'fc1_act')
        fc2 = dense_relu(fc1, self.hidden_2_size, self.hidden_1_size,
            self.phase, add_summary=add_summaries, scope=prefix+'fc2_act')
        out = output_layer(fc2, self.action_shape[0], self.out_init_mag,
            act=tf.tanh, add_summary=add_summaries, scope=prefix+'out_act')
        scaled_out = tf.multiply(out, self.action_bound)

        return obs_in, pos_in, out, scaled_out

    def train(self, obs_in, pos_in, action_gradient, batch_size=1):
        """ Runs training step on actor network. """
        if self.embedding is not None:
            obs_in = self.embedding.create_feed(obs_in)

        self.session.run(self.optimize, feed_dict={
            self.obs_in: obs_in,
            self.pos_in: pos_in,
            self.action_gradient: action_gradient,
            self.phase: 1,
            self.batch_size: batch_size})

    def predict(self, obs_in, pos_in, batch_size=1, add_noise=True, training=1,
        debug=True):
        """ Runs feedforward step on network to predict action. Also returns
            hidden state of LSTM. Debug mode prints output before and after 
            adding noise.
        """
        if self.embedding is not None:
            obs_in = self.embedding.create_feed(obs_in)

        out = self.session.run(self.scaled_out, feed_dict={
                self.obs_in: obs_in,
                self.pos_in: pos_in,
                self.phase: training,
                self.batch_size: batch_size})

        if add_noise:
            if debug: 
                print('Output: ' + str(out))
            out = out + add_noise*self.OU_noise()
            if debug:
                print('Output with noise: ' + str(out))
            out = np.clip(out, -self.action_bound[0], self.action_bound[0])
            if debug:
                print('Output clipped: ' + str(out))

        return out

    def predict_target(self, obs_in, pos_in, batch_size=1):
        """ Runs feedforward step on target network to predict action. """
        if self.embedding is not None:
            obs_in = self.embedding.create_feed(obs_in)

        return self.session.run(self.target_scaled_out, feed_dict={
            self.target_obs_in: obs_in,
            self.target_pos_in: pos_in,
            self.phase: 1,
            self.batch_size: batch_size})

    def update_target_network(self):
        """ Updates target network parameters using Polyak averaging. """
        self.session.run(self.update_target_network_params)

class CriticNetwork(object):
    """ Defines the critic network (Q-network) used in the DDPG algorithm. """

    def __init__(self, session, obs_shape, action_shape, embedding=None):
        """ Initialize critic and target networks and update methods. """
        self.session = session
        self.obs_shape = obs_shape
        self.pos_shape = action_shape # Position component of state
        self.action_shape = action_shape
        self.embedding = embedding # Embedding network (or None)

        self.learning_rate = ddpg_cfg.critic_lr
        self.tau = ddpg_cfg.tau
        self.gamma = ddpg_cfg.gamma
        self.l2_scale = ddpg_cfg.critic_l2_scale

        if self.embedding is not None:
            self.hidden_1_size = ddpg_cfg.hidden_1_size_embedding
            self.hidden_2_size = ddpg_cfg.hidden_2_size_embedding
            self.out_init_mag = ddpg_cfg.out_init_mag_embedding
        else:
            self.filters_per_layer = ddpg_cfg.filters_per_layer # list
            self.stride_per_layer = ddpg_cfg.stride_per_layer # list
            self.kernel_size_per_layer = ddpg_cfg.kernel_size_per_layer # list

            self.hidden_1_size = ddpg_cfg.hidden_1_size_conv
            self.hidden_2_size = ddpg_cfg.hidden_2_size_conv
            self.out_init_mag = ddpg_cfg.out_init_mag_conv
        
        with tf.device(ddpg_cfg.device):
            self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_crt')
            self.phase = tf.placeholder(tf.bool, name='phase_crt')

            # Initialize critic network
            num_vars = len(tf.trainable_variables())
            self.obs_in, self.pos_in, self.action, self.out = \
                self.create_critic_network(add_summaries=True)
            self.network_params = tf.trainable_variables()[num_vars:]

            # Initialize target critic network
            num_vars = len(tf.trainable_variables())
            self.target_obs_in, self.target_pos_in, self.target_action, \
                self.target_out = self.create_critic_network(prefix='tar_')
            self.target_network_params = tf.trainable_variables()[num_vars:]

            # Define target update op
            self.update_target_network_params = \
                [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(self.target_network_params[i], 1.0 - self.tau))
                for i in range(len(self.target_network_params))]

            # Define loss and optimization ops
            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1],
                name='q_pred') # y_i
            self.td_error = tf.square(self.predicted_q_value - self.out)
            self.loss = tf.reduce_mean(self.td_error)

            self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
                minimize(self.loss)

            # Define op for getting gradient of outputs wrt actions
            self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self, prefix='', add_summaries=False):
        """ Constructs the critic network. """
        if self.embedding is not None:
            obs_in, pos_in, to_fc = input_embedding(
                self.embedding,
                self.pos_shape[0],
                self.batch_size,
                self.phase,
                scope=prefix+'input_emb_crt')
        else:
            obs_in, pos_in, to_fc = input_conv(
                self.obs_shape,
                self.pos_shape[0],
                self.filters_per_layer,
                self.stride_per_layer,
                self.kernel_size_per_layer,
                self.batch_size,
                self.phase,
                add_summary=add_summaries,
                scope=prefix+'input_conv_crt')

        action_in = tf.placeholder(shape=[None, self.action_shape[0]],
            dtype=tf.float32, name='act_in')

        fc1 = dense_relu(to_fc, self.hidden_1_size,
            to_fc.get_shape().as_list()[1], self.phase, l2_scale=self.l2_scale,
            add_summary=add_summaries, scope=prefix+'fc1_crt')
        to_fc2 = tf.concat([fc1, action_in], 1)
        fc2 = dense_relu(to_fc2, self.hidden_2_size,
            to_fc2.get_shape().as_list()[1], self.phase, l2_scale=self.l2_scale,
            add_summary=add_summaries, scope=prefix+'fc2_crt')
        out = output_layer(fc2, 1, self.out_init_mag, act=None,
            add_summary=add_summaries, scope=prefix+'out_crt')

        return obs_in, pos_in, action_in, out

    def train(self, obs_in, pos_in, action, predicted_q_value, batch_size=1):
        """ Runs training step on critic network and returns feedforward output.
        """
        if self.embedding is not None:
            obs_in = self.embedding.create_feed(obs_in)

        return self.session.run([self.out, self.optimize], feed_dict={
            self.obs_in: obs_in,
            self.pos_in: pos_in,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.phase: 1,
            self.batch_size: batch_size})

    def predict(self, obs_in, pos_in, action, batch_size=1, training=1):
        """ Runs feedforward step on network to predict Q-value. """
        if self.embedding is not None:
            obs_in = self.embedding.create_feed(obs_in)

        return self.session.run(self.out, feed_dict={
            self.obs_in: obs_in,
            self.pos_in: pos_in,
            self.action: action,
            self.phase: training,
            self.batch_size: batch_size})

    def predict_target(self, obs_in, pos_in, action, batch_size=1):
        """ Runs feedforward step on target network to predict Q-value. """
        if self.embedding is not None:
            obs_in = self.embedding.create_feed(obs_in)

        return self.session.run(self.target_out, feed_dict={
            self.target_obs_in: obs_in,
            self.target_pos_in: pos_in,
            self.target_action: action,
            self.phase: 1,
            self.batch_size: batch_size})

    def action_gradients(self, obs_in, pos_in, actions, batch_size=1):
        """ Returns gradient of outputs wrt actions. """
        if self.embedding is not None:
            obs_in = self.embedding.create_feed(obs_in)

        return self.session.run(self.action_grads, feed_dict={
            self.obs_in: obs_in,
            self.pos_in: pos_in,
            self.action: actions,
            self.phase: 1,
            self.batch_size: batch_size})

    def update_target_network(self):
        """ Updates target network parameters using Polyak averaging. """
        self.session.run(self.update_target_network_params)
