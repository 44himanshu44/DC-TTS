# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function, division

import tensorflow as tf


class CONV(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 size = 1,
                 rate = 1,
                 padding = 'same',
                 dropout_rate = 0,
                 use_bias = True,
                 activation_fn = None):
        super(CONV, self).__init__()
        
        self.size = size
        self.rate = rate
        self.activation_fn = activation_fn
        
        self.params = {"filters": filters, "kernel_size": self.size,
                  "dilation_rate": self.rate, "padding": padding, "use_bias": use_bias,
                  "kernel_initializer": 'glorot_uniform'}
        
        self.conv = tf.keras.layers.Conv1D(**self.params)
        self.dropout = tf.keras.layers.Dropout(rate = dropout_rate)
        self.normalize = tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True)
        
    def call(self, inputs):
        
        tensor = self.conv(inputs)
        tensor = self.normalize(tensor)
        if self.activation_fn is not None:
            tensor = self.activation_fn(tensor)


        tensor = self.dropout(tensor)
        
        return tensor

class HC(tf.keras.layers.Layer):
    def __init__(self,
                filters,
                size = 1,
                rate = 1,
                padding = 'same',
                dropout_rate = 0,
                use_bias = True,
                activation_fn = None):
        super(HC,self).__init__()
        self.size = size,
        self.rate = rate,
        self.activation_fn = activation_fn
        self.params = {'filters':2*filters, 'kernel_size': size,
                      'dilation_rate': rate,'padding': padding, 'use_bias': use_bias,
                      'kernel_initializer':'glorot_uniform'}
        
        self.conv = tf.keras.layers.Conv1D(**self.params)
        self.dropout = tf.keras.layers.Dropout(rate = dropout_rate)
        self.normalize = tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True)
    
    
    def call(self,inputs):
        _inputs = inputs
        
        tensor = self.conv(inputs)
        H1, H2 = tf.split(tensor, 2, axis = -1)
        H1 = self.normalize(H1)
        H2 = self.normalize(H2)
        H1 = tf.nn.sigmoid(H1, 'gate')
        H2 = self.activation_fn(H2, 'info') if self.activation_fn is not None else H2
        tensor = H1*H2 + (1. - H1)* _inputs
        
        tensor = self.dropout(tensor)
        
        return tensor


class CONV_TRANSPOSE(tf.keras.layers.Layer):
    def __init__(self,
                filters,
                size = 3,
                stride = 2,
                padding = 'same',
                dropout_rate = 0,
                use_bias = True,
                activation_fn = None):
        super(CONV_TRANSPOSE,self).__init__()
        self.activation_fn = activation_fn
        self.conv_t = tf.keras.layers.Conv2DTranspose(filters=filters,
                               kernel_size=(1, size),
                               strides=(1, stride),
                               padding=padding,
                               use_bias=use_bias,
                               activation=None,
                               kernel_initializer='glorot_uniform')
        self.normalize = tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True)
        self.dropout = tf.keras.layers.Dropout(rate = dropout_rate)
        
    def call(self,inputs):
        inputs = tf.expand_dims(inputs, 1)
        tensor = self.conv_t(inputs)
        tensor = tf.squeeze(tensor, 1)
        tensor = self.normalize(tensor)

        if self.activation_fn is not None:
            tensor = self.activation_fn(tensor)

        tensor = self.dropout(tensor)
        
        return tensor
        

def Attention(Q, K, V, hp, mononotic_attention=False, prev_max_attentions=None):
    '''
    Args:
      Q: Queries. (B, T/r, d)
      K: Keys. (B, N, d)
      V: Values. (B, N, d)
      mononotic_attention: A boolean. At training, it is False.
      prev_max_attentions: (B,). At training, it is set to None.

    Returns:
      R: [Context Vectors; Q]. (B, T/r, 2d)
      alignments: (B, N, T/r)
      max_attentions: (B, T/r)
    '''
    A = tf.matmul(Q, K, transpose_b=True) * tf.math.rsqrt(tf.dtypes.cast(hp.d, tf.float32))
    if mononotic_attention:  # for inference
        key_masks = tf.sequence_mask(prev_max_attentions, hp.max_N)
        reverse_masks = tf.sequence_mask(hp.max_N - hp.attention_win_size - prev_max_attentions, hp.max_N)[:, ::-1]
        masks = tf.logical_or(key_masks, reverse_masks)
        masks = tf.tile(tf.expand_dims(masks, 1), [1, hp.max_T, 1])
        paddings = tf.ones_like(A) * (-2 ** 32 + 1)  # (B, T/r, N)
        A = tf.where(tf.equal(masks, False), A, paddings)
    A = tf.nn.softmax(A) # (B, T/r, N)
    max_attentions = tf.argmax(A, -1)  # (B, T/r)
    R = tf.matmul(A, V)
    R = tf.concat((R, Q), -1)

    alignments = tf.transpose(A, [0, 2, 1]) # (B, N, T/r)

    return R, alignments, max_attentions

