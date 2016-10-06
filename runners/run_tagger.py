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
import time
### code in this repo
from fergus.algorithms.tagger.utils import rollout
from fergus.configs import global_config
from fergus.algorithms.tagger import Tagger

def run(model, config):

    print("loading data")
    filename_key = "{}_filename".format(config['tag_mode'])
    filepath = os.path.join(config['data_dir'], config[filename_key])
    
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)
    print("data loaded")
    
    if config['fast_mode']: 
        import numpy as np
        data = [data[i] for i in np.random.random_integers(0, len(data), config['fast_mode'])]

    all_stats = []
    with SqliteDict(config['data_store_db'], tablename='indata') as datadb:
        capacitor = {}
        dbkeys = set(map(int,datadb.keys()))
        if len(dbkeys)>0:
            print("{} already done; overwrite={}".format(len(dbkeys), config['overwrite_store']))
        skipped = 0
        for i, datum in tqdm(enumerate(data), total=len(data)):
            if i in dbkeys and not config['overwrite_store']: continue

            if len(datum) > 100: 
                skipped += 1
                continue

            tg = model(datum)
            if len(tg.nodes) == 1:
                print("Degenerate example (#{}). Skipping.".format(i))
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


    print(skipped)


if __name__ == '__main__':
    config = global_config()
    if sys.argv[1].lower() == 'fergusn':
        dbname = "fergus_n_{}_{}".format(config['tag_mode'], config['data_store_db'])
        config['data_store_db'] = os.path.join(config['data_dir'], dbname)
        run(Tagger.fergusn, config)

    elif sys.argv[1].lower() == 'fergusr':
        dbname = "fergus_r_{}_{}".format(config['tag_mode'], config['data_store_db'])
        config['data_store_db'] = os.path.join(config['data_dir'], dbname)                                                
        run(Tagger.fergusr, config)