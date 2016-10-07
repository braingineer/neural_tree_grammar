"""Linearizing routine

Usage:
    run_linearizer.py (fergusr|fergusn) (dev|test) (convolutional|token) 
    run_linearizer.py (-h | --help)

Options:
    fergusr,fergusn             choose the model, fergus recurrent or fergus neuralized
    dev,test                    choose the dataset to run on
    convolution,token           chosen the supertag embedding style
    
    
    
Notes: 
McMahan and Stone
"Syntactic realization with data-driven neural tree grammars" 
published in COLING 2016
"""
from __future__ import absolute_import, print_function, division

import os
import sys
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, '..'))
import fergus

from fergus.algorithms import linearizer
from fergus.models import language_model
from fergus.configs import compose_configs
from sqlitedict import SqliteDict 
from tqdm import tqdm
from docopt import docopt
import numpy as np
import editdistance
import os


def run(config, db_name):
    with SqliteDict(db_name, tablename='tagged_data') as db:
        data = list(db.items())
    with SqliteDict(db_name, tablename='linearized_data') as db:
        finished = set(db.keys())

    data = [datum for datum in data if datum[0] not in finished]

    model = language_model.from_config(config)
    _safety = 2**32
    
    beam = linearizer.decode.beam
    decode_one = linearizer.utils.decode_one
    gendist = linearizer.utils.gendist
    PartialTree = linearizer.partialtree.PartialTree
    editdist = editdistance.eval

    results = {}
    bar = tqdm(total=len(data), desc='partial to trellis to decoding')
    
    for idx, (datum, datum_str, _) in data:
        partial = PartialTree.from_list(datum)
        partial.prep()
        difficulty = partial.measure_difficulty()
        model.logger.debug(str(idx)+' Difficulty: '+str(difficulty))
        if difficulty > 2**35:
            bad_str = "Skipping.  index={}; difficulty={}".format(idx, difficulty)
            model.logger.debug(bad_str)
            bar.update(1)
            continue
        
        seqs, memos = linearizer.dp.run(partial)
        if len(seqs) == 0:
            bad_str = "Failure.  index={}; difficulty={}".format(idx, difficulty)
            model.logger.debug(bad_str)
            bar.update(1)
            continue
        
        datumstr_as_list = datum_str.split(" ")
        datum_len = len(datumstr_as_list)
        
        beam_state, step_decisions, best_idx = beam(memos, model)
        genscores = {}
        edscores = {}
        saving_state = {'datum': datum_str, 
                        'beam_state': beam_state, 
                        'difficulty': difficulty,
                        'generation_distance': [], 
                        'edit_distance': [], 
                        'beam_solutions': [], 
                        'beam_scores': []}
        seen = set()
        for score, beam_idx in beam_state:
            sentence = decode_one(memos, beam_idx)
            assert beam_idx not in seen
            seen.add(beam_idx)
            
            gval = gendist(datumstr_as_list, sentence)
            edval = editdist(datumstr_as_list, sentence)

            saving_state['generation_distance'].append(gval)
            saving_state['edit_distance'].append(edval)
            saving_state['beam_solutions'].append(sentence)
            saving_state['beam_scores'].append(score)
            
        results[idx] = saving_state

        bar.update(1)
        
        if len(results) > 10:            
            with SqliteDict(db_name, tablename='linearized_data') as db:
                db.update(results)
                db.commit()
            results = {}
            
    if len(results) > 0:
        with SqliteDict(db_name, tablename='linearized_data') as db:
            db.update(results)
            db.commit()
        results = {}

    print("Finished.")


if __name__ == '__main__':
    args = docopt(__doc__, version="Linearizer.  publication version. 2016")
    ### the options in docopt doc were mutually exlusive
    data_type = "dev" if args['dev'] else "test"
    embed_type = "convolutional" if args['convolutional'] else "token"
    model_type = "fergusn" if args['fergusn'] else 'fergusr'
    base_name = "{}_{}_{}".format(model_type, embed_type, data_type)
    db_name = base_name+".db"
    
    config = compose_configs(data=data_type, model='language_model')
    
    db_name = os.path.join(config['data_dir'], db_name)
    
    run(config, db_name)