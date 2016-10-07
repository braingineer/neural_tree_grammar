from __future__ import absolute_import, print_function, division

from os.path import join, exists
from tqdm import tqdm
import time
import json
import numpy as np
import sys
from datetime import datetime

import theano

import keras.backend as K
from keras.engine import Model, merge
from keras.layers import Embedding, LSTM, Input, Lambda, \
                         Distribute, RepeatVector, \
                         Reshape, Flatten, Dense, Dropout, \
                         InputLayer, DataLayer, Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import Callback, ProgbarLogger, ModelCheckpoint, ProgbarV2, \
                            LearningRateScheduler, CSVLogger
from keras.regularizers import l2
from keras.utils.visualize_util import plot as kplot

from ikelos.layers import ProbabilityTensor, SoftAttention, Summarize, Fix, LambdaMask, set_name, compose, \
                          BranchLSTM, LastDimDistribute, DynamicEmbedding
from ikelos.data import Vocabulary, VocabManager


from .igor import Igor
from ..common import make_convolutional_embedding, make_logger, make_token_embedding, \
                     make_shallow_convolutional_embedding

try:
    import cPickle as pickle
except:
    import pickle

sys.setrecursionlimit(40000)

concat = lambda layers: merge(layers, mode='concat')
merge_batch = lambda layers: merge(layers, mode='concat', concat_axis=0)

class FergusRModel(object):
    def __init__(self, igor):
        
        now = datetime.now()
        self.run_name = "fergusr_{}mo_{}day_{}hr_{}min".format(now.month, now.day, 
                                                                now.hour, now.minute)
        log_location = join(igor.log_dir, self.run_name+".log")
        self.logger = igor.logger = make_logger(igor, log_location)
        igor.verify_directories()
        self.igor = igor

    @classmethod
    def from_yaml(cls, yamlfile, kwargs=None):
        igor = Igor.from_file(yamlfile)
        igor.prep()
        model = cls(igor)
        model.make(kwargs)
        return model

    @classmethod
    def from_config(cls, config, kwargs=None):
        igor = Igor(config)
        igor.prep()
        model = cls(igor)
        model.make(kwargs)
        return model

    def load_checkpoint_weights(self):
        weight_file = join(self.igor.model_location, 
                           self.igor.saving_prefix,
                           self.igor.checkpoint_weights)
        if exists(weight_file):
            self.logger.info("+ Loading checkpoint weights")
            self.model.load_weights(weight_file, by_name=True)
        else:
            self.logger.warning("- Checkpoint weights do not exist; {}".format(weight_file))



    def plot(self):
        filename = join(self.igor.model_location, 
                        self.igor.saving_prefix, 
                        'model_visualization.png')
        kplot(self.model, to_file=filename)
        self.logger.debug("+ Model visualized at {}".format(filename))


    def make(self, theano_kwargs=None):
        """Construct the Fergus-Recurrent model
        
        Model: 
            Input at time t: 
                - Soft attention over embedded lexemes of children of node_t
                - Embedded lexeme of node_t
            Compute:
                - Inputs are fed into a recurrent tree s.t. hidden states travel down branches
                - node_t's supertag embeddings are retrieved
                - output of recurrent tree at time t is aligned with each supertag vector
                - a vectorized probability function computes a distribution
            Output:
                - Distribution over supertags for node_t
        """
        if self.igor.embedding_type == "convolutional":
            make_convolutional_embedding(self.igor)
        elif self.igor.embedding_type == "token":
            make_token_embedding(self.igor)
        elif self.igor.embedding_type == "shallowconv":
            make_shallow_convolutional_embedding(self.igor)
        else:
            raise Exception("Incorrect embedding type")
        
        
        spine_input_shape = (self.igor.batch_size,
                             self.igor.max_sequence, 
                             self.igor.max_num_supertags) 

        node_input_shape = (self.igor.batch_size, self.igor.max_sequence)

        dctx_input_shape = (self.igor.batch_size, self.igor.max_sequence, self.igor.max_daughter_size)

        E, V = self.igor.word_embedding_size, self.igor.word_vocab_size # for word embeddings
        repeat_N = self.igor.max_num_supertags # for lex
        repeat_D = self.igor.max_daughter_size
        mlp_size = self.igor.mlp_size
        
        ## dropout parameters
        p_emb = self.igor.p_emb_dropout
        p_W = self.igor.p_W_dropout
        p_U = self.igor.p_U_dropout
        w_decay = self.igor.weight_decay
        p_mlp = self.igor.p_mlp_dropout

        #### make layer inputs
        spineset_in = Input(batch_shape=spine_input_shape, name='parent_spineset_in', dtype='int32')
        phead_in = Input(batch_shape=node_input_shape, name='parent_head_input', dtype='int32')
        dctx_in = Input(batch_shape=dctx_input_shape, name='daughter_context_input', dtype='int32')
        topology_in = Input(batch_shape=node_input_shape, name='node_topology', dtype='int32')    

        ##### params
        def predict_params(): 
            return {'output_dim': 1, 
                    'W_regularizer': l2(w_decay), 
                    'activation': 'relu',
                    'b_regularizer':l2(w_decay)}

        ### Layer functions 
        ############# Convert the word indices to vectors
        F_embedword = Embedding(input_dim=V, output_dim=E, mask_zero=True, 
                                W_regularizer=l2(w_decay), dropout=p_emb, name='embedword')
        if self.igor.saved_embeddings is not None:
            print("Loading saved embeddings....")
            F_embedword.initial_weights = [self.igor.saved_embeddings]        
            
        F_probability = ProbabilityTensor(name='predictions', 
                                          dense_function=Dense(**predict_params()))
        ### composition functions 
    
        F_softdaughters = compose(LambdaMask(lambda x, mask: None, name='remove_attention_mask'),
                                  Distribute(SoftAttention(name='softdaughter'), name='distribute_softdaughter'),
                                  F_embedword)

        F_align = compose(Distribute(Dropout(p_mlp)),
                           Distribute(Dense(mlp_size, activation='relu')),
                           concat)
                           
        F_rtn = compose(RepeatVector(repeat_N, axis=2, name='repeattree'),
                        BranchLSTM(self.igor.rtn_size, name='recurrent_tree1', return_sequences=True))

        F_predict = compose(Distribute(F_probability, name='distribute_probability'),
                            Distribute(Dropout(p_mlp)),  ### need a separate one because the 'concat' is different for the two situations
                            LastDimDistribute(Dense(mlp_size, activation='relu')),
                            concat)

        ############################ new ###########################

        dctx = F_softdaughters(dctx_in)
        parent = F_embedword(phead_in)
        #node_context = F_align([parent, dctx])
        #import pdb
        #pdb.set_trace()

        ### put into tree
        aligned_node = F_align([parent, dctx])
        node_context = F_rtn([aligned_node, topology_in])
            
        parent_spines = self.igor.F_embedspine(spineset_in)
        ### get probability
        predictions = F_predict([node_context, parent_spines])

        ##################
        ### make model
        ##################
        self.model = Model(input=[dctx_in, phead_in, topology_in, spineset_in], 
                           output=predictions,
                           preloaded_data=self.igor.preloaded_data)

        ##################
        ### compile model
        ##################
        optimizer = Adam(self.igor.LR, clipnorm=self.igor.max_grad_norm, 
                                       clipvalue=self.igor.grad_clip_threshold)
        theano_kwargs = theano_kwargs or {}
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=['accuracy'], **theano_kwargs)


        if self.igor.from_checkpoint:
            self.load_checkpoint_weights()
        elif not self.igor.in_training:
            raise Exception("No point in running this without trained weights")
            

    def train(self):
        train_data = self.igor.train_gen(forever=True)
        dev_data = self.igor.dev_gen(forever=True)
        N = self.igor.num_train_samples
        E = self.igor.num_epochs
        # generator, samplers per epoch, number epochs
        callbacks = [ProgbarV2(3, 10)]
        
        checkpoint_fp = join(self.igor.model_location,
                             self.igor.saving_prefix,
                             self.igor.checkpoint_weights)
        self.logger.info("+ Model Checkpoint: {}".format(checkpoint_fp))
        
        callbacks += [ModelCheckpoint(filepath=checkpoint_fp, verbose=1, save_best_only=True)]
        callbacks += [LearningRateScheduler(lambda epoch: self.igor.LR * 0.9**(epoch))]
        
        csv_location = join(self.igor.log_dir, self.run_name+".csv")
        callbacks += [CSVLogger(csv_location)]
        self.model.fit_generator(generator=train_data, samples_per_epoch=N, nb_epoch=E,
                                 callbacks=callbacks, verbose=1,
                                 validation_data=dev_data,
                                 nb_val_samples=self.igor.num_dev_samples)

    def debug(self):
        dev_data = self.igor.dev_gen(forever=False)
        X,Y = next(dev_data)
        self.model.predict_on_batch(X)
        #self.model.evaluate_generator(dev_data, self.igor.num_dev_samples)


    def profile(self, num_iterations=1):
        train_data = self.igor.train_gen(forever=True)
        dev_data = self.igor.dev_gen(forever=True)
        # generator, samplers per epoch, number epochs
        callbacks = [ProgbarV2(1, 10)]
        self.logger.debug("+ Beginning the generator")
        self.model.fit_generator(generator=train_data, 
                                 samples_per_epoch=self.igor.batch_size*10, 
                                 nb_epoch=num_iterations,
                                 callbacks=callbacks, 
                                 verbose=1,
                                 validation_data=dev_data,
                                 nb_val_samples=self.igor.batch_size)
        self.logger.debug("+ Calling theano's pydot print.. this might take a while")
        theano.printing.pydotprint(self.model.train_function.function, 
                                   outfile='theano_graph.png',
                                   var_with_name_simple=True,
                                   with_ids=True)
        self.logger.debug("+ Calling keras' print.. this might take a while")
        self.plot("keras_graph.png")
        #self.model.profile.print_summary()

    def __call__(self, data):
        if self.model is None:
            raise Exception("model not instantiated yet; please call make()")
        assert isinstance(data, list)
        B = data[0].shape[0]
        return self.model.predict(data, batch_size=B)