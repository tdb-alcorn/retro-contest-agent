import tensorflow as tf
import numpy as np
from typing import Union, Tuple
from functools import reduce
from agents.agent import Agent
from agents.config import env, supervised
cfg = supervised['conv']
from agents.supervised.supervised_agent import Supervised


# This agent only implements learn, load and save. It is intended for offline learning only.
# This particular instance implements a convolutional auto-encoder for transferring to
# the conv rnn rl2 agent.
class SupervisedConv(Supervised):
    def _checkpoint_name(self):
        return "checkpoints/{}.ckpt".format(self.name)

    def _restore_graph(self, _input:tf.Tensor, training:tf.Tensor, name:str='SupervisedConv'):
        '''Returns input and output tensors.'''
        checkpoint_name = self._checkpoint_name()
        # self.output = tf.train.import_meta_graph('{}.meta'.format(checkpoint_name), input_map={
        tf.train.import_meta_graph('{}.meta'.format(checkpoint_name), input_map={
            'input': _input,
            'training': training,
        })
        # }, return_elements=[
        #     'embedding',
        # ])
        graph = tf.get_default_graph()
        self.output = graph.get_tensor_by_name('embedding')
        # self.input, self.output, self.training = graph.get_collection('interface')

    def __init__(self,
        _input:Union[tf.Tensor, None]=None,
        training:Union[tf.Tensor, None]=None,
        *args,
        name:str='SupervisedConv',
        component:bool=False,
        **kwargs,
        ):
        if _input is None:
            _input = tf.zeros([1, *env["state_shape"]], dtype=tf.float32)
        if training is None:
            training = tf.constant(False)
        self.name = name
        self._component = component
        self.losses = list()
        if self._component:
            self._restore_graph(_input, training, name=name)
        else:
            self._build_graph(_input, training, *args, name=name, **kwargs)

    def _build_graph(self,
        _input: tf.Tensor,
        training: tf.Tensor,
        state_shape=env["state_shape"],
        learning_rate=cfg['learning_rate'],
        name:str='SupervisedConv',
        ):
        # if _input is None:
        # _input = tf.zeros([1, *env["state_shape"]], dtype=tf.float32)
        # if training is None:
        # training = tf.constant(False)
        with tf.variable_scope(name):
            # Inputs
            # self._input = tf.placeholder(tf.float32, [None, *state_shape], name='state')
            # self.input = tf.placeholder_with_default(_input, [None, *state_shape], name='input')
            self.input = tf.reshape(tf.Variable(_input, validate_shape=False, trainable=False, name='input'), [-1, *state_shape])
            # self._training = tf.placeholder(tf.bool, name='training')
            # self.training = tf.placeholder_with_default(training, [], name='training')
            self.training = tf.reshape(tf.Variable(training, validate_shape=False, trainable=False, name='training'), [])
            self.dropout_rate = cfg['dropout']

            # Convolutional layers
            self.conv_layers = [self.input]
            for i in range(1, len(cfg['layers'])):
                prev_layer = self.conv_layers[i-1]
                layer_cfg = cfg['layers'][i]
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

            self.embedding = tf.layers.dense(self.conv_out, cfg['embedding'], activation=tf.nn.relu, name='embedding')

            self.deconv_input = tf.layers.dense(self.embedding, reduce(lambda x,y:x*y, conv_out_shape), activation=tf.nn.relu)

            # Deconvolutional layers
            self.deconv_layers = [tf.reshape(self.deconv_input, [-1, *conv_out_shape])]
            layer_idx = 0
            for i in range(len(cfg['layers'])-1, -1, -1):
                prev_layer = self.deconv_layers[layer_idx]
                layer_cfg = cfg['layers'][i]
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

            # Interface
            for var in [self.input, self.output, self.training]:
                tf.add_to_collection('interface', var)

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
        ):
        checkpoint_name = self._checkpoint_name()
        if not hasattr(self, 'saver'):
            train_vars = tf.trainable_variables(scope=self.name)
            self.saver = tf.train.Saver(train_vars)
        try:
            self.saver.restore(sess, checkpoint_name)
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
            print("SupervisedConv.load: checkpoint file not found, skipping load")

    def save(self,
        sess:tf.Session,
        ):
        checkpoint_name = self._checkpoint_name()
        if not hasattr(self, 'saver'):
            train_vars = tf.trainable_variables(scope=self.name)
            self.saver = tf.train.Saver(train_vars)
        self.saver.save(sess, checkpoint_name)
