"""Supertagging routine

Usage:
    run_tagger.py (fergusr|fergusn) (dev|test) (convolutional|token) 
    run_tagger.py (-h | --help)

Options:
    fergusr,fergusn             choose the model, fergus recurrent or fergus neuralized
    dev,test                    choose the dataset to run on
    convolution,token           chosen the supertag embedding style
    
    
    
Notes: 
McMahan and Stone
"Syntactic realization with data-driven neural tree grammars" 
published in COLING 2016
"""

from __future__ import print_function, division, absolute_import
import pickle
import sys
import os
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, '..'))
import fergus

### additional packages needed
from tqdm import tqdm
from sqlitedict import SqliteDict
from docopt import docopt
import time
### code in this repo
from fergus.algorithms.tagger.utils import rollout
from fergus.configs import compose_configs
from fergus.algorithms.tagger import Tagger

def run(model_factory, config, db_name):
    
    db_name = os.path.join(config['data_dir'], db_name)
    data_file = os.path.join(config['data_dir'], config['data_file'])

    with open(data_file, 'rb') as fp:
        data = pickle.load(fp)
    
    if config['fast_mode']: 
        import numpy as np
        data = [data[i] for i in np.random.random_integers(0, len(data), config['fast_mode'])]

    all_stats = []
    with SqliteDict(db_name, tablename='tagged_data') as datadb:
        capacitor = {}
        dbkeys = set(map(int,datadb.keys()))
        if len(dbkeys)>0:
            print("{} already done.  will be overwriting={}".format(len(dbkeys), config['overwrite_store']))
        skipped = 0
        for i, datum in tqdm(enumerate(data), total=len(data), ncols=10):
            if i in dbkeys and not config['overwrite_store']: continue
            if len(datum) > 100: 
                skipped += 1
                continue

            tg = model_factory(datum, config)
            if len(tg.nodes) == 1:
                print("Degenerate example (#{}). Skipping.".format(i))
                skipped += 1
                continue
            start = time.time()
            try:
                tg.run()
                capacitor[i] = (tg.root.to_list(), str(rollout(datum)), tg.experiment_stats)
            except Exception as e:
                print("EXCEPTION: {}".format(e))
                print("sys.exc_info(): {}".format(sys.exc_info()))
            
            if config['verbose']:
                for k, v in sorted(tg.stats().items()):
                    print("{:<25} =  {:>0.3f}".format(k, v))
                    
            if (i+1) % 20 == 0:
                datadb.update(capacitor)
                datadb.commit()
                capacitor = {}
                
        if len(capacitor) > 0:
            datadb.update(capacitor)
            datadb.commit()
            capacitor = {}


    print("{} examples were skipped because too long (>100) or too short (==1)".format(skipped))


if __name__ == '__main__':
    args = docopt(__doc__, version="Supertagger.  publication version. 2016")
    ### the options in docopt doc were mutually exlusive
    data_type = "dev" if args['dev'] else "test"
    embed_type = "convolutional" if args['convolutional'] else "token"
    model_type = "fergusn" if args['fergusn'] else 'fergusr'
    base_name = "{}_{}_{}".format(model_type, embed_type, data_type)
    db_name = base_name+".db"
    conf_name = base_name+".conf"
    model_factory = {'fergusn': Tagger.fergusn, 'fergusr': Tagger.fergusr}[model_type]
    
    config = compose_configs(data=data_type, model=model_type, embedding=embed_type)
    
    run(model_factory, config, db_name)