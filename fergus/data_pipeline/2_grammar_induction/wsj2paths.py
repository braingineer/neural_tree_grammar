"""Process the WSJ treebank to paths and possible surface forms

Usage:
    wsj2paths.py convert <config_file> [-s]
    wsj2paths.py (-h | --help)

    convert <config_file>         all necessary configuration should be in here
                                  an example of the config is below. 
    -s --surface                  save the surface form from the paths
    -h --help                     show this



Example (Cohn, Chiang, etc) Config:
---
wsj_root_dir: /research/data/LDC/treebank3/treebank_3/parsed/mrg/wsj/
splits: 
  - 02-21
  - 22-22
  - 23-23
save_name: chiang_wsj
---

Example (Mikolov, etc) Config:
---
wsj_root_dir: /research/data/LDC/treebank3/treebank_3/parsed/mrg/wsj/
splits: 
  - 00-20
  - 21-22
  - 23-24
save_name: mikolov_wsj
---           
"""
from __future__ import print_function
from collections import defaultdict, Counter
from os.path import join
from docopt import docopt
from baal.nlp.induce.tree_enrichment import get_trees, string2cuts
from baal.science.gist import common
from baal.nlp.structures import DerivationTree, ElementaryTree
import baal
import pickle
import yaml
from tqdm import tqdm, trange
import sys
import glob

from ..common import config

def parse_splits(config):
    splits = config.splits
    _F = lambda i: "{:02}".format(i)
    F = lambda l,u: [_F(i) for i in range(l,u)]
    splits = [x.split("-") for x in splits]
    splits = [F(int(l), int(u)+1) for l,u in splits]
    return {k:v for k,v in zip(["train", "dev", "test"],splits)}

def safe_single_parse(parse):
    try:
        _, sf = common.parse2predicates(parse)
        tree, subtrees = baal.nlp.induce.tree_enrichment.parse2derivation(parse)
        assert sf == str(tree).replace("_", " ")
        return True, (tree.pre_order_features(), sf)
    except Exception as e:
        return False, (parse, e)

def run():
    splits = parse_splits(config)
    base_dir = os.path.join(config.wsj_root_dir, "final/[0-9]*.mrg")
    files = {'weird': "weird_stuff_{}.txt".format(config['save_name']),
             'good_save': config['save_name']+'_{}.pkl',
             'bad_save': config['save_name']+'_bad_{}.pkl',
             'surface_save': config['save_name']+'_flat_{}.txt'}
    files = {k:"data/"+v for k,v in files.items()}
    open(files['weird'],"w").close()

    sucbar = tqdm(desc='decimating trees', unit=' trees', position=0, leave=True, total=0)
    for data_type, sections in splits.items():
        good_paths = []
        surface_forms = []
        bad_parses = []
        for section in sections:
            filepath = os.path.join(config.wsj_root_dir, 'final', '{}.mrg'.format(section))
            print("+ Processing {}".format(filepath))
            for treestr in get_trees(filepath, True):
                success, data = safe_single_parse(treestr)
                if success:
                    good_paths.append(data[0])
                    if save_surface:
                        surface_forms.append(data[1])
                else:
                    sucbar.n -= 1
                    bad_parses.append(data)
                sucbar.total += 1
                sucbar.update(1)

        with open(files['good_save'].format(data_type),"w") as fp:
            pickle.dump(good_paths, fp)

        with open(files['bad_save'].format(data_type), "w") as fp:
            pickle.dump(bad_parses, fp)

        with open(files['surface_save'].format(data_type), 'w') as fp:
            fp.write('\n'.join(surface_forms))