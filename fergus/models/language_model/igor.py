from __future__ import print_function, division, absolute_import
import yaml
from baal.utils import loggers
from tqdm import tqdm, trange
import numpy as np
import itertools
import time
import sys
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from keras.utils.np_utils import to_categorical
import os
from os.path import join
from . import utils
import baal
import gist
from ikelos.data import VocabManager, Vocabulary
import sys

try:
    input = raw_input
except:
    pass
try:
    import cPickle as pickle
except:
    pass

class Igor(object):
    def __init__(self, config):
        self.__dict__.update(config)

    @classmethod
    def from_file(cls, config_file):
        with open(config_file) as fp:
            config = yaml.load(fp)
        return cls(config)

    @property
    def num_train_batches(self):
        return len(self.train_data)//self.batch_size

    @property
    def num_dev_batches(self):
        return len(self.dev_data)//self.batch_size
        
    @property
    def num_test_batches(self):
        return len(self.test_data)//self.batch_size
        
    @property
    def num_train_samples(self):
        return self.num_train_batches // 3 * self.batch_size
        
    @property
    def num_dev_samples(self):
        return self.num_dev_batches * self.batch_size
        
    @property
    def num_test_samples(self):
        return self.num_test_batches * self.batch_size

    def serve_sentence(self, data):
        for data_i in np.random.choice(len(data), len(data), replace=False):
            in_X = np.zeros(self.max_sequence_len)
            out_Y = np.zeros(self.max_sequence_len, dtype=np.int32)
            bigram_data = zip(data[data_i][0:-1], data[data_i][1:])
            for datum_j,(datum_in, datum_out) in enumerate(bigram_data):
                in_X[datum_j] = datum_in
                out_Y[datum_j] = datum_out
            yield in_X, out_Y

    def serve_batch(self, data):
        dataiter = self.serve_sentence(data)
        V = self.vocab_size
        S = self.max_sequence_len
        B = self.batch_size

        while dataiter:
            in_X = np.zeros((B, S), dtype=np.int32)
            out_Y = np.zeros((B, S, V), dtype=np.int32)
            next_batch = list(itertools.islice(dataiter, 0, self.batch_size))
            if len(next_batch) < self.batch_size:
                raise StopIteration
            for d_i, (d_X, d_Y) in enumerate(next_batch):
                in_X[d_i] = d_X
                out_Y[d_i] = to_categorical(d_Y, V)
                
            yield in_X, out_Y

    def _data_gen(self, data, forever=True):
        working = True
        while working:
            for batch in self.serve_batch(data):
                yield batch
            working = working and forever
        
    def dev_gen(self, forever=True):
        return self._data_gen(self.dev_data, forever)

    def train_gen(self, forever=True):
        return self._data_gen(self.train_data, forever)            
        
    def test_gen(self):
        return self._data_gen(self.test_data, False)
    
    def prep(self):
        save_dir = join(self.model_location, self.saving_prefix+'/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.vocman_file = os.path.join(self.data_dir, self.vocman_file)
        self.vocabs = VocabManager.from_file(self.vocman_file)

        if '<START>' not in self.vocabs.words:
            self.vocabs.words.unfreeze()
            self.vocabs.words.add('<START>')
            self.vocabs.words.add('<END>')
            self.vocabs.words.add('<NUMBER>')
            fdir = "/".join(self.vocman_file.split("/")[:-1])
            self.vocabs.save(fdir, self.vocman_file.split("/")[-1])
            self.vocabs.words.freeze(True)

        if not self.vocabs.words._frozen:
            self.vocabs.words.freeze(True)
            fdir = "/".join(self.vocman_file.split("/")[:-1])
            self.vocabs.save(fdir, self.vocman_file.split("/")[-1])

        self.train_fp = os.path.join(self.data_dir, self.train_filepath)
        self.dev_fp = os.path.join(self.data_dir, self.dev_filepath)
        self.test_fp = os.path.join(self.data_dir, self.test_filepath)
        self.embeddings_file = os.path.join(self.data_dir, self.embeddings_file)
        
        if self.load_data:
            train_raw =  utils.load_dataset(self.train_fp, False, self.max_sentence_length)       
            self.train_data = utils.convert(train_raw, self.vocabs.words)
    
            dev_raw = utils.load_dataset(self.dev_fp, False, self.max_sentence_length)
            self.dev_data = utils.convert(dev_raw, self.vocabs.words)
            
            test_raw = utils.load_dataset(self.test_fp, False, self.max_sentence_length)
            self.test_data = utils.convert(test_raw, self.vocabs.words)


            max_seq = max([len(sent) for sent in self.test_data+self.train_data+self.dev_data])
            self.max_sequence_len = max_seq
        else:
            self.max_sequence_len = self.max_sentence_length
        self.vocab_size = len(self.vocabs.words)

        ### current implementation assumes an embedding
        self.embeddings = np.load(self.embeddings_file)
        