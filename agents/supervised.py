import tensorflow as tf
import numpy as np
from functools import reduce
from agents.agent import Agent
from agents.config import env, deep_q
cfg = deep_q['conv_recurrent_rl2']


# This agent only implements learn, load and save. It is intended for offline learning only.
# This particular instance implements a convolutional auto-encoder for transferring to
# the conv rnn rl2 agent.
# TODO: Factor out the autoencoder bits to make a general supervised training class.
class SupervisedConv(Agent):
    def __init__(self,
        state_shape=env["state_shape"],
        learning_rate=cfg['learning_rate'],
        name='SupervisedConv',
        ):
        self.name = name
        self.losses = list()
        self.checkpoint_name = "checkpoints/{}.ckpt".format(name)
        with tf.variable_scope(name):
            # Inputs
            self.input = tf.placeholder(tf.float32, [None, *state_shape], name='state')
            self.training = tf.placeholder(tf.bool, name='training')
            self.dropout_rate = cfg['dropout']

            # Convolutional layers
            self.conv_layers = [self.input]
            for i in range(1, len(cfg['conv_layers'])):
                prev_layer = self.conv_layers[i-1]
                layer_cfg = cfg['conv_layers'][i]
                # TODO Make type an enum
                if layer_cfg['type'] == 'conv2d':
                    layer = tf.layers.conv2d(
                        prev_layer,
                        layer_cfg['filters'],
                        layer_cfg['kernel_size'],
                        strides=layer_cfg['strides'],
                        padding='same',
                        activation=tf.nn.relu,
                    )
                elif layer_cfg['type'] == 'max_pool2d':
                    layer = tf.layers.max_pooling2d(prev_layer, layer_cfg['pool_size'], layer_cfg['strides'], padding='same')
                
                # Handle dropout
                if hasattr(layer_cfg, 'dropout') and layer_cfg['dropout']:
                    layer = tf.layers.dropout(layer, rate=self.dropout_rate, training=self.training)
                self.conv_layers.append(layer)

            conv_out_shape = self.conv_layers[-1].get_shape().as_list()[1:]
            self.conv_out = tf.layers.flatten(self.conv_layers[-1])

            self.embedding = tf.layers.dense(self.conv_out, cfg['embedding'], activation=tf.nn.relu)

            self.deconv_input = tf.layers.dense(self.embedding, reduce(lambda x,y:x*y, conv_out_shape), activation=tf.nn.relu)

            # Deconvolutional layers
            self.deconv_layers = [tf.reshape(self.deconv_input, [-1, *conv_out_shape])]
            layer_idx = 0
            for i in range(len(cfg['conv_layers'])-1, -1, -1):
                prev_layer = self.deconv_layers[layer_idx]
                layer_cfg = cfg['conv_layers'][i]
                if layer_cfg['type'] == 'conv2d':
                    layer = tf.layers.conv2d_transpose(
                        prev_layer,
                        layer_cfg['filters'],
                        layer_cfg['kernel_size'],
                        strides=layer_cfg['strides'],
                        padding='same',
                        activation=tf.nn.relu,
                    )
                # elif layer_cfg['type'] == 'max_pool2d':
                #     dims = [d * layer_cfg['strides'] for d in prev_layer.get_shape().as_list()[1:3]]
                #     # import pdb; pdb.set_trace()
                #     layer = tf.image.resize_images(prev_layer, dims, method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
                self.deconv_layers.append(layer)
                layer_idx += 1
            
            self.output = tf.layers.conv2d_transpose(
                self.deconv_layers[-1],
                3,  # RGB
                5,
                strides=2,
                padding='same',
                activation=None,
            )
            self.loss = tf.reduce_mean(tf.square(self.output - self.input))
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def learn(self,
        sess:tf.Session,
        states:np.array,
        actions:np.array,
        rewards:np.array,
        next_states:np.array,
        episode_ends:np.array
        ) -> float:
        loss, _ = sess.run([self.loss, self.output], feed_dict={
            self.input: states,
            self.training: True,
        })
        self.losses.append(loss)
        return loss

    def load(self,
        sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        train_vars = tf.trainable_variables(scope=self.name)
        saver = tf.train.Saver(train_vars)
        try:
            saver.restore(sess, self.checkpoint_name)
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
            print("deep_q_agent.load: checkpoint file not found, skipping load")

    def save(self,
        sess:tf.Session,
        # saver:tf.train.Saver,
        ):
        train_vars = tf.trainable_variables(scope=self.name)
        saver = tf.train.Saver(train_vars)
        saver.save(sess, self.checkpoint_name)

    # Not implemented
    def act(self,
        sess:tf.Session,
        state:np.array,
        train:bool,
        ) -> np.array:
        pass

    # Not implemented
    def step(self,
        sess:tf.Session,
        state:np.array,
        action:np.array,
        reward:float,
        next_state:np.array,
        done:bool
        ):
        pass