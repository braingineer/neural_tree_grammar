
from __future__ import print_function, division

import sys
import time
from copy import copy, deepcopy
from os.path import join, exists
from collections import Counter
from math import log
import itertools
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from ikelos.data import Vocabulary

from keras.layers import LSTM, Dense, Embedding, Distribute, Dropout, Input
from keras.callbacks import Callback, ProgbarLogger, ModelCheckpoint, ProgbarV2, LearningRateScheduler
from keras.utils.generic_utils import Progbar
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.engine import Model
import keras.backend as K


try:
    input = raw_input
except:
    pass    
try:
    import cPickle as pickle
except:
    pass
    
from ..common import make_logger
from .igor import Igor


sys.setrecursionlimit(40000)
def compose(*layers):
    def func(x):
        out = x
        for layer in layers[::-1]:
            out = layer(out)
        return out
    return func

class LanguageModel(object):
    def __init__(self, igor):
        now = datetime.now()
        self.run_name = "fergusr_{}mo_{}day_{}hr_{}min".format(now.month, now.day, 
                                                                now.hour, now.minute)
        log_location = join(igor.log_dir, self.run_name+".log")
        self.logger = igor.logger = make_logger(igor, log_location)
        self.igor = igor
        
    @classmethod
    def from_config(cls, config):
        igor = Igor(config)
        igor.prep()
        model = cls(igor)
        model.make()
        return model


    def make(self):

        B = self.igor.batch_size
        R = self.igor.rnn_size
        S = self.igor.max_sequence_len
        V = self.igor.vocab_size
        E = self.igor.embedding_size
        emb_W = self.igor.embeddings.astype(K.floatx())
        
        ## dropout parameters
        p_emb = self.igor.p_emb_dropout
        p_W = self.igor.p_W_dropout
        p_U = self.igor.p_U_dropout
        p_dense = self.igor.p_dense_dropout
        w_decay = self.igor.weight_decay

        def embedding_parameters():
            return {"W_regularizer": l2(w_decay),
                    "weights": [emb_W],
                    "mask_zero": True,
                    "dropout": p_emb}
                    
        def sequence_parameters():
            return {"return_sequences": True,
                    "dropout_W": p_W,
                    "dropout_U": p_U,
                    "U_regularizer": l2(w_decay),
                    "W_regularizer": l2(w_decay)}
        def predict_parameters():
            return {"activation": 'softmax',
                    "W_regularizer": l2(w_decay),
                    "b_regularizer": l2(w_decay)}
                    
        F_embed = Embedding(V, E, **embedding_parameters())
        F_seq1 = LSTM(R, **sequence_parameters())
        F_seq2 = LSTM(R*int(1/p_dense), **sequence_parameters())
        F_drop = Dropout(p_dense)
        F_predict = Distribute(Dense(V, **predict_parameters()))
        

        words_in = Input(batch_shape=(B,S), dtype='int32')
        predictions = compose(F_predict,
                              F_drop,
                              F_seq2,
                              F_drop,
                              F_seq1,
                              F_embed)(words_in)
        
        #self.F_p = K.Function([words_in, K.learning_phase()], predictions)

        optimizer = Adam(self.igor.LR, clipnorm=self.igor.max_grad_norm, 
                                       clipvalue=self.igor.max_grad_value)
        self.model = Model(input=[words_in], 
                           output=[predictions])
        self.model.compile(loss='categorical_crossentropy', 
                           optimizer=optimizer, 
                           metrics=['accuracy', 'perplexity'])

        if self.igor.from_checkpoint:
            self.load_checkpoint_weights()
            
    def load_checkpoint_weights(self):
        weight_file = join(self.igor.model_location, 
                           self.igor.saving_prefix,
                           self.igor.checkpoint_weights)
        if exists(weight_file):
            self.logger.info("+ Loading checkpoint weights")
            self.model.load_weights(weight_file, by_name=True)
        else:
            self.logger.warning("- Checkpoint weights do not exist; {}".format(weight_file))

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
        callbacks += [LearningRateScheduler(lambda epoch: self.igor.LR * 0.95 ** (epoch % 15))]
        self.model.fit_generator(generator=train_data, samples_per_epoch=N, nb_epoch=E,
                                 callbacks=callbacks, verbose=1,
                                 validation_data=dev_data,
                                 nb_val_samples=self.igor.num_dev_samples)

    def test(self, num_samples=None):
        num_samples = num_samples or 100
        test_data = self.igor.test_gen()
        out = self.model.evaluate_generator(test_data, num_samples)
        try: 
            for o, label in zip(out, self.model.metric_names):
                print("{}: {}".format(o, label))
        except Exception as e:
            print("some sort of error.. {}".format(e))
            import pdb
            pdb.set_trace()

    def format_sentence(self, sentence):
        ''' turn into indices here '''
        if not isinstance(sentence, list):
            sentence = sentence.split(" ")
        sentence = [self.igor.vocabs.words[w] for w in sentence]

        in_X = np.zeros(self.max_sequence_len)
        out_Y = np.zeros(self.max_sequence_len, dtype=np.int32)
        bigram_data = zip(sentence[0:-1], sentence[1:])
        for datum_j,(datum_in, datum_out) in enumerate(bigram_data):
            in_X[datum_j] = datum_in
            out_Y[datum_j] = datum_out
        return in_X, out_Y
    
    def eval_sentence(self, sentence):
        X, y = self.format_sentence(sentence)
        yout = self.F_p([X[None,:]]+[0.])
        yout = yout[0]
        return X, y, yout

    def sample(self):
        L = self.igor.train_vocab.lookup
        for dev_datum in self.igor.dev_gen():
            X, y = dev_datum # X.shape = (b,s); y.shape = (b,s,V)
            Px = self.model.predict_proba(X) # Px.shape = (b,s,V)
            for i in range(X.shape[0]): 
                w_in = []
                w_true = []
                w_tprob = []
                w_pprob = []
                w_pred = []
                for j in range(X.shape[1]):
                    if L(X[i][j]) == "<MASK>": continue
                    w_in.append(L(X[i][j]))
                    w_true.append(L(y[i][j].argmax()))
                    w_pred.append(L(Px[i][j].argmax()))
                    w_tprob.append(Px[i][j][y[i][j].argmax()])
                    w_pprob.append(Px[i][j].max())

                n = max([len(w) for w in w_true+w_pred]) + 6

                for wt,wi,pwt,wp,pwp in zip(w_true, w_in, w_tprob, w_pred,w_pprob):
                    s = "|\t{:0.6f}\t|{:>%d} => {:<%d}|{:^%d}|\t{:0.6f}\t|" % (n,n,n)
                    print(s.format(pwt, wi,wt, wp, pwp))
            
                perp = 2**(-sum([log(p,2) for p in w_tprob]) / (len(w_tprob)-1))
                print("Per word perplexity of sentence: {:0.3f}".format(perp))

                prompt = input("<enter to continue, y to enter pdb, exit to exit>")
                if prompt == "y":
                    import pdb
                    pdb.set_trace()
                elif prompt == "exit":
                    import sys
                    sys.exit(0)

    def examine(self):
        L = self.igor.train_vocab.lookup
        sent_ppls = []
        sent_lls = []
        count = 0
        for dev_datum in self.igor.dev_gen(False):
            X, y = dev_datum # X.shape = (b,s); y.shape = (b,s,V)
            Px = self.model.predict_proba(X) # Px.shape = (b,s,V)
            for i in range(X.shape[0]): 
                word_probs = []
                for j in range(X.shape[1]):
                    if L(X[i][j]) == "<MASK>": continue    
                    word_probs.append(Px[i][j][y[i][j].argmax()])            
                perp = 2**(-sum([log(p,2) for p in word_probs]) / (len(word_probs)-1))
                sent_lls.append(-sum([log(p,2) for p in word_probs]))
                count += len(word_probs)
                sent_ppls.append(perp)
        with open("ppls.pkl", "w") as fp:
            pickle.dump(sent_ppls, fp)

        print("PERPLEXITIES")
        print("Mean: {}".format(np.mean(sent_ppls)))
        print("Median: {}".format(np.median(sent_ppls)))


        
        ent = sum(sent_lls) / (count-1.0)
        print("from sent lls and then calculated after: {:0.5f}".format(2**ent))

        
        plot = plt.hist(sent_ppls, bins=20)
        plt.show()
