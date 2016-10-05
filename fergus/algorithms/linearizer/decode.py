from __future__ import absolute_import, print_function, division
from . import utils
from fergus.models import language_model
from sqlitedict import SqliteDict as SD
from tqdm import tqdm
import numpy as np
import editdistance
import os
import sys

######## different ways to decode

def w2v(words, igor):
    vec = np.zeros((igor.batch_size, igor.max_sequence_len), dtype=np.int32)
    for i, word in enumerate(words):
        vec[0, i] = igor.vocabs.words[word]
    return vec

def step(model, sentence, choices, verbose=0):
    vec = w2v(sentence)
    idx = len(sentence) - 1# with just <start>, len is 1. we want post over start, so subtract 1
    probs = model.model.predict(vec)[0,idx,:]
    best = 0.
    best_word = ""
    best_idx = -1
    decisions = []
    for i, word in enumerate(choices):
        word_val = probs[model.igor.vocabs.words[word]]
        decisions.append((word, word_val))
        if word_val > best:
            best = word_val
            best_word = word
            best_idx = i
    #if verbose:
    #    print("selecting {}".format(best_word))
    #    print("-------")
    return sentence + [best_word], best_idx, decisions

def mle(memos, verbose=0):
    bwd, fwd = memos
    cur_idx = -1
    sentence = ['<START>']
    decisions = []
    while cur_idx in fwd:
        next_idxs = fwd[cur_idx]
        choices = [bwd[poss_idx][1] for poss_idx in next_idxs]
        sentence, selection_idx, decision = step(model, sentence, choices, verbose)
        decisions.append(decision)
        cur_idx = list(next_idxs)[selection_idx] 
    if verbose: print(' '.join(sentence))
    return cur_idx, decisions


def vectorize(sentences, igor):
    out = np.zeros((igor.batch_size, igor.max_sequence_len), dtype=np.int32)
    tp1_idx = np.zeros((igor.batch_size), dtype=np.int32)
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            out[i,j] = igor.vocabs.words[word]
        tp1_idx[i] = j ## will be len(sentence) - 1
    return out, tp1_idx

def beam(memos, model, verbose=0):
    igor = model.igor
    assert '<START>' in igor.vocabs.words._mapping
    beam_size = igor.batch_size
    sentences = [['<START>']]
    step_indices = [-1]
    path_scores = [0.]
    bwd, fwd = memos
    decisions = []

    while all([step_idx in fwd for step_idx in step_indices]):
        utils.logprint(1, 'taking a beam step')
        decisions.append([])
        next_indices = [fwd[idx] for idx in step_indices]
        next_words = [[bwd[idx_tp1][1] for idx_tp1 in idx_set] for idx_set in next_indices]
        batch, pred_indices = vectorize(sentences, igor)
        predictions = model.model.predict(batch)[np.arange(beam_size),pred_indices,:]
        
        ranking = []
        for master_index in range(min(beam_size, len(step_indices))):
            posterior = predictions[master_index]
            words_tp1 = next_words[master_index] 
            score = path_scores[master_index]
            idx_tp1 = next_indices[master_index] 
            sentence = sentences[master_index]
            word_vec = np.array([igor.vocabs.words[word] for word in words_tp1])
            local_posterior = posterior[word_vec] #/ posterior[word_vec].sum()
            word_scores = -np.log(local_posterior)
            possible_sentences = [sentence + [word] for word in words_tp1]
            word_tuples = zip(word_scores+score, idx_tp1, possible_sentences)
            ranking.extend(word_tuples)
            for word, word_score in zip(words_tp1, word_scores):
                decisions[-1].append((word, word_score))

            
        ranking = sorted(ranking)[:beam_size]
        step_indices = [path_idx for _, path_idx, _ in ranking]
        path_scores = [score for score, _, _ in ranking]
        sentences = [sentence for _, _, sentence in ranking]
    
    out_ = zip(path_scores, step_indices)
    out = [(score,idx) for score, idx in out_ if idx not in fwd]
    if len(out) == 0:
        out = out_ ### just in case
        print("out was length 0. weird.")
        
    best_score, best_idx = min(out)        
    if verbose:
        print('best: '+' '.join(decode_one(memos, best_idx)))
    return out, decisions, best_idx
