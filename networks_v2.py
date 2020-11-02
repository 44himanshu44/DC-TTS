# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from layers import *
import tensorflow as tf

class TEXTENC(tf.keras.Model):
    def __init__(self):
        super(TEXTENC, self).__init__()
    
        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
        self.embedding = tf.keras.layers.Embedding(len(hp.vocab),hp.e,embeddings_initializer = initializer,mask_zero = True)
        self.conv_relu = CONV(filters = 2*hp.d,size = 1,rate = 1,
                                    dropout_rate = hp.dropout_rate,activation_fn = tf.nn.relu)
        self.conv_1 = CONV(filters = 2*hp.d,size = 1,rate = 1,
                                    dropout_rate = hp.dropout_rate)
        self.hc_1 = HC(filters = 2*hp.d,size = 3,rate = 3**0,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_2 = HC(filters = 2*hp.d,size = 3,rate = 3**1,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_3 = HC(filters = 2*hp.d,size = 3,rate = 3**2,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_4 = HC(filters = 2*hp.d,size = 3,rate = 3**3,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_5 = HC(filters = 2*hp.d,size = 3,rate = 3**0,
                                   dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_6 = HC(filters = 2*hp.d,size = 3,rate = 3**1,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_7 = HC(filters = 2*hp.d,size = 3,rate = 3**2,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_8 = HC(filters = 2*hp.d,size = 3,rate = 3**3,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_9 = HC(filters = 2*hp.d,size = 3,rate = 1,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_10 = HC(filters = 2*hp.d,size = 3,rate = 1,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_11 = HC(filters = 2*hp.d,size = 1,rate = 1,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_12 = HC(filters = 2*hp.d,size = 1,rate = 1,
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
    def call(self, inputs):
        embed = self.embedding(inputs)
        tensor = self.conv_relu(embed)
        tensor = self.conv_1(tensor)
        tensor = self.hc_1(tensor)
        tensor = self.hc_2(tensor)
        tensor = self.hc_3(tensor)
        tensor = self.hc_4(tensor)
        tensor = self.hc_5(tensor)
        tensor = self.hc_6(tensor)
        tensor = self.hc_7(tensor)
        tensor = self.hc_8(tensor)
        tensor = self.hc_9(tensor)
        tensor = self.hc_10(tensor)
        tensor = self.hc_11(tensor)
        tensor = self.hc_12(tensor)
        
        k,v = tf.split(tensor,2,-1)
        
        return k,v


class AUDIOENC(tf.keras.Model):
    def __init__(self):
        super(AUDIOENC,self).__init__()
        self.conv_relu_1 = CONV(filters = hp.d,size = 1,rate = 1,padding ='causal',
                        dropout_rate = hp.dropout_rate,activation_fn = tf.nn.relu)
        self.conv_relu_2 = CONV(filters = hp.d,size = 1,rate = 1,padding ='causal',
                        dropout_rate = hp.dropout_rate,activation_fn = tf.nn.relu)
        self.conv_relu_3 = CONV(filters = hp.d,size = 1,rate = 1,padding ='causal',
                        dropout_rate = hp.dropout_rate,activation_fn = tf.nn.relu)
        self.hc_1 = HC(filters = hp.d,size = 3,rate = 3**0,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_2 = HC(filters = hp.d,size = 3,rate = 3**1,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_3 = HC(filters = hp.d,size = 3,rate = 3**2,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_4 = HC(filters = hp.d,size = 3,rate = 3**3,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_5 = HC(filters = hp.d,size = 3,rate = 3**0,padding = 'causal',
                                   dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_6 = HC(filters = hp.d,size = 3,rate = 3**1,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_7 = HC(filters = hp.d,size = 3,rate = 3**2,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_8 = HC(filters = hp.d,size = 3,rate = 3**3,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_9 = HC(filters = hp.d,size = 3,rate = 3,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_10 = HC(filters = hp.d,size = 3,rate = 3,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        
    def call(self, inputs):
        tensor = self.conv_relu_1(inputs)
        tensor = self.conv_relu_2(tensor)
        tensor = self.conv_relu_3(tensor)
        tensor = self.hc_1(tensor)
        tensor = self.hc_2(tensor)
        tensor = self.hc_3(tensor)
        tensor = self.hc_4(tensor)
        tensor = self.hc_5(tensor)
        tensor = self.hc_6(tensor)
        tensor = self.hc_7(tensor)
        tensor = self.hc_8(tensor)
        tensor = self.hc_9(tensor)
        tensor = self.hc_10(tensor)
        
        return tensor


class AUDIODEC(tf.keras.Model):
    def __init__(self):
        super(AUDIODEC, self).__init__()
        self.conv_1 = CONV(filters = hp.d,size = 1,rate = 1,padding ='causal',
                        dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_1 = HC(filters = hp.d,size = 3,rate = 3**0,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_2 = HC(filters = hp.d,size = 3,rate = 3**1,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_3 = HC(filters = hp.d,size = 3,rate = 3**2,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_4 = HC(filters = hp.d,size = 3,rate = 3**3,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_5 = HC(filters = hp.d,size = 3,rate = 1,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_6 = HC(filters = hp.d,size = 3,rate = 1,padding = 'causal',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.conv_relu_1 = CONV(filters = hp.d,size = 1,rate = 1,padding ='causal',
                        dropout_rate = hp.dropout_rate,activation_fn = tf.nn.relu)
        self.conv_relu_2 = CONV(filters = hp.d,size = 1,rate = 1,padding ='causal',
                        dropout_rate = hp.dropout_rate,activation_fn = tf.nn.relu)
        self.conv_relu_3 = CONV(filters = hp.d,size = 1,rate = 1,padding ='causal',
                        dropout_rate = hp.dropout_rate,activation_fn = tf.nn.relu)
        self.conv_2 = CONV(filters = hp.n_mels,size = 1,rate = 1,padding ='causal',
                        dropout_rate = hp.dropout_rate,activation_fn = None)
        self.activation_ = tf.nn.sigmoid
        
    def call(self,inputs):
        tensor = self.conv_1(inputs)
        tensor = self.hc_1(tensor)
        tensor = self.hc_2(tensor)
        tensor = self.hc_3(tensor)
        tensor = self.hc_4(tensor)
        tensor = self.hc_5(tensor)
        tensor = self.hc_6(tensor)
        tensor = self.conv_relu_1(tensor)
        tensor = self.conv_relu_2(tensor)
        tensor = self.conv_relu_3(tensor)
        logits = self.conv_2(tensor)
        Y = self.activation_(logits)
        
        return logits, Y
        
class SSRN(tf.keras.Model):
    def __init__(self):
        super(SSRN, self).__init__()
        self.conv_1 = CONV(filters = hp.c,size = 1,rate = 1,padding ='same',
                        dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_1 = HC(filters = hp.c,size = 3,rate = 3**0,padding = 'same',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_2 = HC(filters = hp.c,size = 3,rate = 3**1,padding = 'same',
                                   dropout_rate = hp.dropout_rate,activation_fn = None)
        
        self.conv_t1 = CONV_TRANSPOSE(filters = hp.c,dropout_rate = hp.dropout_rate)
        self.conv_t2 = CONV_TRANSPOSE(filters = hp.c,dropout_rate = hp.dropout_rate)
        self.hc_3 = HC(filters = hp.c,size = 3,rate = 3**0,padding = 'same',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_4 = HC(filters = hp.c,size = 3,rate = 3**1,padding = 'same',
                                   dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_5 = HC(filters = hp.c,size = 3,rate = 3**0,padding = 'same',
                                    dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_6 = HC(filters = hp.c,size = 3,rate = 3**1,padding = 'same',
                                   dropout_rate = hp.dropout_rate,activation_fn = None)
        self.conv_2 = CONV(filters = 2*hp.c,size = 1,rate = 1,padding ='same',
                        dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_7 = HC(filters = 2*hp.c,size = 3,rate = 1,padding = 'same',
                                   dropout_rate = hp.dropout_rate,activation_fn = None)
        self.hc_8 = HC(filters = 2*hp.c,size = 3,rate = 1,padding = 'same',
                                   dropout_rate = hp.dropout_rate,activation_fn = None)
        self.conv_3 = CONV(filters = 1+hp.n_fft//2,size = 1,rate = 1,padding ='same',
                        dropout_rate = hp.dropout_rate,activation_fn = None)
        self.conv_4_relu = CONV(filters = 1+hp.n_fft//2,size = 1,rate = 1,padding ='same',
                        dropout_rate = hp.dropout_rate,activation_fn = tf.nn.relu)
        self.conv_5_relu = CONV(filters = 1+hp.n_fft//2,size = 1,rate = 1,padding ='same',
                        dropout_rate = hp.dropout_rate,activation_fn = tf.nn.relu)
        self.conv_6 = CONV(filters = 1+hp.n_fft//2,size = 1,rate = 1,padding ='same',
                        dropout_rate = hp.dropout_rate,activation_fn = None)
        self.activation_ = tf.nn.sigmoid
    
    def call(self,inputs):
        tensor = self.conv_1(inputs)
        tensor = self.hc_1(tensor)
        tensor = self.hc_2(tensor)
        tensor = self.conv_t1(tensor)
        tensor = self.hc_3(tensor)
        tensor = self.hc_4(tensor)
        tensor = self.conv_t2(tensor)
        tensor = self.hc_5(tensor)
        tensor = self.hc_6(tensor)
        tensor = self.conv_2(tensor)
        tensor = self.hc_7(tensor)
        tensor = self.hc_8(tensor)
        tensor = self.conv_3(tensor)
        tensor = self.conv_4_relu(tensor)
        tensor = self.conv_5_relu(tensor)
        logits = self.conv_6(tensor)
        Z = self.activation_(logits)
        
        return logits,Z
        
        
        