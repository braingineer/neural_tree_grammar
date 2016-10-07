from __future__ import absolute_import, print_function, division

import baal
import theano
import time
import json
import numpy as np
from os.path import join, exists


#### need ikelos library; see braingineer/ikelos
from ikelos.data import Vocabulary, VocabManager
from ikelos.layers import ProbabilityTensor, SoftAttention, Summarize, Fix, LambdaMask, \
                          set_name, compose, LastDimDistribute, DynamicEmbedding

#### need my version of keras; see braingineer/keras
import keras.backend as K
from keras.layers import Embedding, LSTM, Input, Lambda, \
                         Distribute, RepeatVector, \
                         Reshape, Flatten, Dense, Dropout, \
                         InputLayer, Convolution2D, MaxPooling2D
from keras.engine import Model, merge
from keras.optimizers import Adam
from keras.callbacks import Callback, ProgbarLogger, ModelCheckpoint, ProgbarV2, LearningRateScheduler
from keras.regularizers import l2
from keras.utils.visualize_util import plot as kplot

### project imports
from .igor import Igor
from ..common import make_convolutional_embedding, make_logger, make_token_embedding


try:
    import cPickle as pickle
except:
    import pickle
import sys;
sys.setrecursionlimit(40000)

concat = lambda layers: merge(layers, mode='concat')
merge_batch = lambda layers: merge(layers, mode='concat', concat_axis=0)


def traverse_nodes(layer, from_name="top", down=True):
    #print("- At {} from {}".format(layer.name, from_name))
    nodes = layer.inbound_nodes
    #print("\t+{} inbound nodes".format(len(nodes)))
    mask_cache = {}
    for node in layer.inbound_nodes:
        #print("\t + {} layers feeding into node".format(len(node.inbound_layers)))
        #print("\t + {} shapes feeding into node".format("; ".join(map(str, node.input_shapes))))
        #print("\t + ({}) masks feeding into node".format("; ".join(["yes" if mask is not None else "no" 
        #                                                          for mask in node.input_masks])))
        for i, mask in enumerate(node.input_masks):
            if mask is not None:
                mask_cache['{}.in.mask.{}'.format(layer.name, i)] = mask
        for layer_down in node.inbound_layers:
            mask_cache.update(traverse_nodes(layer_down, layer.name))
    return mask_cache

class FergusNModel(object):
    def __init__(self, igor):
        
        now = datetime.now()
        self.run_name = "fergusn_{}mo_{}day_{}hr_{}min".format(now.month, now.day, 
                                                                now.hour, now.minute)
        log_location = os.path.join(igor.log_dir, self.run_name+".log")
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
        '''Make the model and compile it. 

        Igor's config options control everything.  

        Arg:
            theano_kwargs as dict for debugging theano or submitting something custom
        '''
        
        if self.igor.embedding_type == "convolutional":
            make_convolutional_embedding(self.igor)
        elif self.igor.embedding_type == "token":
            make_token_embedding(self.igor)
        else:
            raise Exception("Incorrect embedding type")
        
        B = self.igor.batch_size
        spine_input_shape = (B, self.igor.max_num_supertags) 
        child_input_shape = (B, 1)
        parent_input_shape = (B, 1)

        E, V = self.igor.word_embedding_size, self.igor.word_vocab_size # for word embeddings
        
        repeat_N = self.igor.max_num_supertags # for lex
        mlp_size = self.igor.mlp_size
        
        ## dropout parameters
        p_emb = self.igor.p_emb_dropout
        p_W = self.igor.p_W_dropout
        p_U = self.igor.p_U_dropout
        w_decay = self.igor.weight_decay
        p_mlp = self.igor.p_mlp_dropout

        def predict_params():
            return {'output_dim': 1, 
                   'W_regularizer': l2(w_decay), 
                   'activation': 'relu',
                   'b_regularizer':l2(w_decay)}

        dspineset_in = Input(batch_shape=spine_input_shape, name='daughter_spineset_in', dtype='int32')
        pspineset_in = Input(batch_shape=spine_input_shape, name='parent_spineset_in', dtype='int32')
        dhead_in = Input(batch_shape=child_input_shape, name='daughter_head_input', dtype='int32')
        phead_in = Input(batch_shape=parent_input_shape, name='parent_head_input', dtype='int32')
        dspine_in = Input(batch_shape=child_input_shape, name='daughter_spine_input', dtype='int32')
        inputs = [dspineset_in, pspineset_in, dhead_in, phead_in, dspine_in]

        ### Layer functions 
        ############# Convert the word indices to vectors
        F_embedword = Embedding(input_dim=V, output_dim=E, mask_zero=True, 
                                W_regularizer=l2(w_decay), dropout=p_emb)

        if self.igor.saved_embeddings is not None:
            self.logger.info("+ Cached embeddings loaded")
            F_embedword.initial_weights = [self.igor.saved_embeddings]

        ###### Prediction Functions
        ## these functions learn a vector which turns a tensor into a matrix of probabilities

        ### P(Parent supertag | Child, Context)
        F_parent_predict = ProbabilityTensor(name='parent_predictions', 
                                          dense_function=Dense(**predict_params()))
        ### P(Leaf supertag)
        F_leaf_predict = ProbabilityTensor(name='leaf_predictions', 
                                           dense_function=Dense(**predict_params()))

        ###### Network functions.  
        ##### Input word, correct its dimensions (basically squash in a certain way)
        F_singleword = compose(Fix(),
                               F_embedword)
        ##### Input spine, correct diemnsions, broadcast across 1st dimension
        F_singlespine = compose(RepeatVector(repeat_N),
                                Fix(),
                                self.igor.F_embedspine)
        ##### Concatenate and map to a single space
        F_alignlex = compose(RepeatVector(repeat_N),
                             Dropout(p_mlp), 
                             Dense(mlp_size, activation='relu', name='dense_align_lex'),
                             concat)

        F_alignall = compose(Distribute(Dropout(p_mlp), name='distribute_align_all_dropout'),
                              Distribute(Dense(mlp_size, activation='relu', name='align_all_dense'), name='distribute_align_all_dense'),
                              concat)
        F_alignleaf = compose(Distribute(Dropout(p_mlp*0.66), name='distribute_leaf_dropout'),  ### need a separate oen because the 'concat' is different for the two situations
                              Distribute(Dense(mlp_size, activation='relu', name='leaf_dense'), name='distribute_leaf_dense'),
                              concat)

        ### embed and form all of the inputs into their components 
        ### note: spines == supertags. early word choice, haven't refactored. 
        leaf_spines = self.igor.F_embedspine(dspineset_in)
        pspine_context = self.igor.F_embedspine(pspineset_in)
        dspine_single = F_singlespine(dspine_in)

        dhead = F_singleword(dhead_in)
        phead = F_singleword(phead_in)

        ### combine the lexical material        
        lexical_context = F_alignlex([dhead, phead])

        #### P(Parent Supertag | Daughter Supertag, Lexical Context)
        ### we know the daughter spine, want to know the parent spine
        ### size is (batch, num_supertags)
        parent_problem = F_alignall([lexical_context, dspine_single, pspine_context])

        ### we don't have the parent, we just have a leaf
        leaf_problem = F_alignleaf([lexical_context, leaf_spines])

        parent_predictions = F_parent_predict(parent_problem)
        leaf_predictions = F_leaf_predict(leaf_problem)
        predictions = [parent_predictions, leaf_predictions]

        theano_kwargs = theano_kwargs or {}
        ## make it quick so i can load in the weights. 
        self.model = Model(input=inputs, output=predictions, 
                           preloaded_data=self.igor.preloaded_data, **theano_kwargs)

        #mask_cache = traverse_nodes(parent_prediction)
        #desired_masks = ['merge_3.in.mask.0']
        #self.p_tensor = K.function(inputs+[K.learning_phase()], [parent_predictions, F_parent_predict.inbound_nodes[0].input_masks[0]])
        
        if self.igor.from_checkpoint:
            self.load_checkpoint_weights()
        elif not self.igor.in_training:
            raise Exception("No point in running this without trained weights")
            
            
        if not self.igor.in_training:
            expanded_children = RepeatVector(repeat_N, axis=2)(leaf_spines)        
            expanded_parent = RepeatVector(repeat_N, axis=1)(pspine_context)
            expanded_lex = RepeatVector(repeat_N, axis=1)(lexical_context) # axis here is arbitary; its repeating on 1 and 2, but already repeated once
            huge_tensor = concat([expanded_lex, expanded_children, expanded_parent])
            densely_aligned = LastDimDistribute(F_alignall.get(1).layer)(huge_tensor)
            output_predictions = Distribute(F_parent_predict, force_reshape=True)(densely_aligned)

            primary_inputs = [phead_in, dhead_in, pspineset_in, dspineset_in]
            leaf_inputs = [phead_in, dhead_in, dspineset_in]
            
            self.logger.info("+ Compiling prediction functions")
            self.inner_func = K.Function(primary_inputs+[K.learning_phase()], output_predictions)
            self.leaf_func = K.Function(leaf_inputs+[K.learning_phase()], leaf_predictions)    
            try:
                self.get_ptensor = K.function(primary_inputs+[K.learning_phase()], 
                                              [output_predictions, ])
            except:
                import pdb
                pdb.set_trace()
        else:

            optimizer = Adam(self.igor.LR, clipnorm=self.igor.max_grad_norm, 
                                            clipvalue=self.igor.grad_clip_threshold)

            theano_kwargs = theano_kwargs or {}
            self.model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                                metrics=['accuracy'], **theano_kwargs)
                                
        #self.model.save("here.h5")

    def likelihood_function(self, inputs):
        if self.igor.in_training:
            raise Exception("Not in testing mode; please fix the config file")
        return self.inner_func(tuple(inputs) + (0.,))

    def leaf_function(self, inputs):
        if self.igor.in_training:
            raise Exception("Not in testing mode; please fix the config file")
        return self.leaf_func(tuple(inputs) + (0.,))


    def train(self):
        replacers = {"daughter_predictions":"child",
                     "parent_predictions":"parent",
                     "leaf_predictions":"leaf"}
        train_data = self.igor.train_gen(forever=True)
        dev_data = self.igor.dev_gen(forever=True)
        N = self.igor.num_train_samples 
        E = self.igor.num_epochs
        # generator, samplers per epoch, number epochs
        callbacks = [ProgbarV2(3, 10, replacers=replacers)]
        checkpoint_fp = join(self.igor.model_location,
                             self.igor.saving_prefix,
                             self.igor.checkpoint_weights)
        self.logger.info("+ Model Checkpoint: {}".format(checkpoint_fp))
        callbacks += [ModelCheckpoint(filepath=checkpoint_fp, verbose=1, save_best_only=True)]
        callbacks += [LearningRateScheduler(lambda epoch: self.igor.LR * 0.9)]
        csv_location = os.path.join(self.igor.log_dir, self.run_name+".csv")
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
