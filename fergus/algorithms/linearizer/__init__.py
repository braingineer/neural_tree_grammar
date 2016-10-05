from __future__ import absolute_import
from . import partialtree, astar, dp, states, decode

from sqlitedict import SqliteDict as SD
import numpy as np
import time
from . import utils
import sys

def run(tree_list, *args, **kwargs):
    partial = partialtree.PartialTree.from_list(tree_list)
    partial.prep()
    results = astar.run(partial, *args, **kwargs)
    return results

def run_dp(tree_list, *args, **kwargs):
    partial = partialtree.PartialTree.from_list(tree_list)
    partial.prep()
    print(partial.measure_difficulty())
    try:
        results = dp.run(partial, *args, **kwargs)
    except utils.MemoryException as me:
        print(me.msg)
        results = []
    return results, partial.measure_difficulty()

def run_tests():
    utils.level = 1
    db_path = '/home/cogniton/research/code/gist/gist/alternate_models/fergus616_dev_R.db'
    with SD(db_path, tablename='indata') as db:
        data = list(db.items())

    n = len(data)
    r_times = []
    s_times = []
    start = time.time()
    last = start
    import pprint
    for idx in np.random.choice(np.arange(len(data)), 10, False):
        pprint.PrettyPrinter(indent=2).pprint(data[idx][1][0])
        run(data[idx][1][0])
        now = time.time()
        r_times.append(now-start)
        s_times.append(now - last)
        last = now
        print("[tic. r:{:0.2f} toc. s:{:0.2f}] \r".format(now-start, now-last))
        print(data[idx][1][1])
        sys.stdout.flush()
    print("finished.  {:0.2f} seconds".format(time.time()-start))
