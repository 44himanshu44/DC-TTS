# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function, division

import tensorflow as tf


def embed(inputs, vocab_size, num_units, zero_pad=True):
    '''Embeds a given tensor. 
    
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
    '''
    initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
    outputs = tf.keras.layers.Embedding(vocab_size, num_units, input_length=10,embeddings_initializer = initializer,mask_zero = True)(inputs)

    return outputs


def normalize(inputs):
    '''Applies layer normalization that normalizes along the last axis.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. The normalization is over the last dimension.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    outputs = tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True)(inputs)
    return outputs


def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387

    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not num_units:
        num_units = inputs.get_shape()[-1]

   
    H = tf.keras.layers.Dense(units=num_units, activation='relu', name="dense1")(inputs)
    T = tf.keras.layers.Dense(units=num_units, activation='sigmoid',
                        bias_initializer = tf.constant_initializer(-1.0), name="dense2")(inputs)
    outputs = H * T + inputs * (1 - T)
    return outputs

def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           dropout_rate=0,
           use_bias=True,
           activation_fn=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      dropout_rate: A float of [0, 1].
      use_bias: A boolean.
      activation_fn: A string.
      training: A boolean. If True, dropout is applied.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    
    if padding.lower() == "causal":
        # pre-padding for causality
        pad_len = (size - 1) * rate  # padding size
        inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
        padding = "valid"
    
    
    if filters is None:
        filters = inputs.get_shape().as_list()[-1]

    params = {"filters": filters, "kernel_size": size,
              "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
              "kernel_initializer": 'glorot_uniform'}

    tensor = tf.keras.layers.Conv1D(**params)(inputs)
    tensor = normalize(tensor)
    if activation_fn is not None:
        tensor = activation_fn(tensor)
    

    tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor)

    return tensor

def hc(inputs,
       filters=None,
       size=1,
       rate=1,
       padding="SAME",
       dropout_rate=0,
       use_bias=True,
       activation_fn=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      activation_fn: A string.
      training: A boolean. If True, dropout is applied.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    _inputs = inputs
   
    if padding.lower() == "causal":
        # pre-padding for causality
        pad_len = (size - 1) * rate  # padding size
        inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_len, 0], [0, 0]])
        padding = "valid"

    if filters is None:
        filters = inputs.get_shape().as_list()[-1]


    params = {"filters": 2*filters, "kernel_size": size,
              "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
              "kernel_initializer": 'glorot_uniform'}

    tensor =  tf.keras.layers.Conv1D(**params)(inputs)
    H1, H2 = tf.split(tensor, 2, axis=-1)
    H1 = normalize(H1)
    H2 = normalize(H2)
    H1 = tf.nn.sigmoid(H1, "gate")
    H2 = activation_fn(H2, "info") if activation_fn is not None else H2
    tensor = H1*H2 + (1.-H1)*_inputs

    tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor)

    return tensor

def conv1d_transpose(inputs,
                     filters=None,
                     size=3,
                     stride=2,
                     padding='same',
                     dropout_rate=0,
                     use_bias=True,
                     activation=None):
    '''
        Args:
          inputs: A 3-D tensor with shape of [batch, time, depth].
          filters: An int. Number of outputs (=activation maps)
          size: An int. Filter size.
          rate: An int. Dilation rate.
          padding: Either `same` or `valid` or `causal` (case-insensitive).
          dropout_rate: A float of [0, 1].
          use_bias: A boolean.
          activation_fn: A string.
          training: A boolean. If True, dropout is applied.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A tensor of the shape with [batch, time*2, depth].
        '''
    
    if filters is None:
        filters = inputs.get_shape().as_list()[-1]
    inputs = tf.expand_dims(inputs, 1)
    tensor = tf.keras.layers.Conv2DTranspose(filters=filters,
                               kernel_size=(1, size),
                               strides=(1, stride),
                               padding=padding,
                               use_bias=use_bias,
                               activation=None,
                               kernel_initializer='glorot_uniform')(inputs)
    
    tensor = tf.squeeze(tensor, 1)
    tensor = normalize(tensor)
    if activation is not None:
        tensor = activation(tensor)

    tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor)

    return tensor





