# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import codecs
import os
import re
import unicodedata

import numpy as np
import tensorflow as tf
from hyperparams import Hyperparams as hp
from utils_v2 import *


def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode=="train":
        if "LJ" in hp.data:
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines:
                fname, _, text = line.strip().split("|")

                fpath = os.path.join(hp.data, "wavs", fname + ".wav")
                fpaths.append(fpath)

                text = text_normalize(text) + "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32))

            return fpaths, text_lengths, texts
        else: # nick or kate
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines:
                fname, _, text, is_inside_quotes, duration = line.strip().split("|")
                duration = float(duration)
                if duration > 10. : continue

                fpath = os.path.join(hp.data, fname)
                fpaths.append(fpath)

                text += "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32))

        return fpaths, text_lengths, texts

    else: # synthesize on unseen test text.
        # Parse
        lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
        texts = np.zeros((len(sents), hp.max_N), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data() # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.B

       

        hp.prepro = False
        if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                mel = "mels/{}".format(fname.replace("wav", "npy"))
                mag = "mags/{}".format(fname.replace("wav", "npy"))
                return fname, np.load(mel), np.load(mag)

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = load_spectrograms(fpaths)

        
        return texts, mel, mag, fname, num_batch


        
# load all lists
L, mels, mags, fnames, num_batch = get_batch()

# create a list of lists
data_text = [[L[i],mels[i],mags[i], fnames[i],len(L[i]),num_batch] for i in range(len(L))]

# sort the data in the ascending order of text length
data_text.sort(key = lambda x : x[4])

sorted_with_len = [(sent[0],sent[1],sent[2],sent[3],sent[4],sent[5]) for sent in data_text if sent[4] > 10]

# run this command multiple times to store chunks of data
sorted_with_len_13099 = []
for i in range(12000,13100):
    sorted_with_len_13099.append(sorted_with_len[i])
with open('data/sorted_with_len_13099','wb') as f:
    pickle.dump(sorted_with_len_13099,f)



    
