from __future__ import absolute_import, print_function, division
from .utils import decode_node, decode_spine, encode_spine, cartesian_iterator
from ...models import fergus_neuralized, fergus_recurrent
from baal.structures import DerivationTree
from . import astar_core as astar
from baal.induce import enrich as ten
from copy import deepcopy
from . import tagger_node
from . import astar_state
import numpy as np
import sys
import time

def singleton(cls):
    cached = {}
    def getinstance(*args, **kwargs):
        if cls not in cached:
            cached[cls] = cls(*args, **kwargs)
        return cached[cls]
    return getinstance

class Tagger(object):
    def __init__(self, path, model):
        self.model = model
        self.igor = model.igor
        self.root = model.Node.from_path(path, self)
        self.root.annotate()
        self.nodes = self.root.flat
        self.runtime = -1
        
    @classmethod
    def fergusn(cls, path):
        tg = cls(path, MNeuralized())
        return tg

    @classmethod
    def fergusr(cls, path):
        tg = cls(path, MRecurrent())
        return tg

    def stats(self, raw=False):
        stats = self.root.full_comparison_stats()
        if not raw:
            stats = {k:float(sum(v))/len(v) for k,v in stats.items()}
        return stats
    
    @property
    def experiment_stats(self):
        stats = {}
        stats['before_astar'] = self.classify_only_stats
        stats['after_astar'] = self.stats(True)        
        stats['length'] = len(self.root.flat)
        stats['time'] = self.runtime
        return stats

    def before_stats(self):
        for node in self.root.flat.values():
            assert node.best_index is not None
            node.set_supertag(node.best_index)
        self.classify_only_stats = self.stats(True)
        
    
            
    def run(self, verbose=0):
        start_time = time.time()
        self.model.compute_likelihoods(self.root)
        topside_score = self.root.compute_downside()
        self.root.cache_upside(topside_score)
        self.before_stats()
        astar.run(self)
        self.model.Node.reset()
        self.runtime = time.time() - start_time

    def get_partial(self):
        return PartialTree(self.root) 

    def make_tree(self, spine_id, head_word):
        return self.igor.reconstruct(spine_id, head_word)


@singleton
class MNeuralized(object):
    Node = tagger_node.ZeroNode
    State = astar_state.ZeroState

    def __init__(self):
        self.model = fergus_neuralized.globally_set_model()
        self.model.igor.compute_affordances()
        self.igor = self.model.igor

    def serve_treelet_batches(self, root, batch_size):
        capacitor = []
        for treelet in root.treelets:
            capacitor.append(treelet)
            if len(capacitor) == batch_size:
                yield capacitor
                capacitor = []
        if len(capacitor) > 0:
            yield capacitor

    def serve_leaf_batches(self, root, batch_size):
        capacitor = []
        for leaf in root.leaves:
            capacitor.append(leaf)
            if len(capacitor) == batch_size:
                yield capacitor
                capacitor = [] 
        if len(capacitor) > 0:
            yield capacitor

    def padded_spines(self, word_id):
        spineset = self.igor.get_spineset(word_id)
        n = self.igor.spineset_size - len(spineset)
        assert spineset.ndim == 1
        return np.pad(spineset, (0,n), 'constant')

    def format_lhood_batch(self, treelet_batch):    
        igor = self.igor
        X_parent = np.zeros((igor.batch_size,1))
        X_child = np.zeros((igor.batch_size,1))
        X_parentset = np.zeros((igor.batch_size, igor.spineset_size))
        X_childset = np.zeros((igor.batch_size, igor.spineset_size))

        for d_i, (parent, daughter) in enumerate(treelet_batch):
            p_id = igor.vocabs.words[parent.head]
            d_id = igor.vocabs.words[daughter.head]
            X_parent[d_i] = p_id
            X_child[d_i] = d_id
            X_parentset[d_i] = self.padded_spines(p_id)
            X_childset[d_i] = self.padded_spines(d_id)


        return X_parent, X_child, X_parentset, X_childset

        
    def format_leaf_batch(self, leaf_batch):
        igor = self.igor
        X_leaf = np.zeros((igor.batch_size,1))
        X_parent = np.zeros((igor.batch_size, 1))
        X_leafspineset = np.zeros((igor.batch_size, igor.spineset_size))
        for d_i, leaf in enumerate(leaf_batch):
            leaf_id = igor.vocabs.words[leaf.head]
            parent_id = igor.vocabs.words[leaf.parent.head]
            X_leaf[d_i] = leaf_id
            X_parent[d_i] = parent_id
            X_leafspineset[d_i] = self.padded_spines(leaf_id)

        return X_parent, X_leaf, X_leafspineset

    def compute_likelihoods(self, root):
        igor = self.igor
        for treelet_batch in self.serve_treelet_batches(root, igor.batch_size):
            lhoods = self.model.likelihood_function(list(self.format_lhood_batch(treelet_batch)))
            for li, (parent, daughter) in enumerate(treelet_batch):
                parent.add_distribution(daughter.id, lhoods[li])
                
        for leaf_batch in self.serve_leaf_batches(root, self.igor.batch_size):
            prior_values = self.model.leaf_function(list(self.format_leaf_batch(leaf_batch)))
            for li, leaf in enumerate(leaf_batch):
                leaf.distribution = -1 * np.log(prior_values[li])

@singleton
class MRecurrent(object):
    Node = tagger_node.RNode
    State = astar_state.RState

    def __init__(self):
        self.model = fergus_recurrent.globally_set_model()
        self.model.igor.compute_affordances()
        self.igor = self.model.igor

    def make_sequence(self, root):
        pmap = {-1:0}
        nmap = {}
        out = []
        # child, parent id, head id, daughter ids
        for i, (child, pid, hid, dids) in enumerate(root.sequence):
            out.append((pmap[pid], hid, dids))
            pmap[child.id] = i
            nmap[i] = child
        return out, pmap, nmap

    def format_tree(self, tree_sequence):
        igor = self.igor
        N_spine = igor.max_num_spines
        N_seq = igor.max_sequence
        N_dctx = igor.max_daughter_size

        X_dcontext = np.zeros((1, N_seq, N_dctx))
        X_spineset = np.zeros((1, N_seq, N_spine))
        X_head = np.zeros((1, N_seq))
        X_topo = np.zeros((1, N_seq))

        for n_i, (p_idx, head_idx, dot_idxs) in enumerate(tree_sequence):
            X_head[0, n_i] = head_idx
            X_topo[0, n_i] = p_idx
            X_dcontext[0, n_i][:len(dot_idxs)] = dot_idxs
            for i, spine in enumerate(igor.get_spineset(head_idx)):
                X_spineset[0, n_i][i] = spine

        return [X_dcontext, X_head, X_topo, X_spineset]

    def compute_likelihoods(self, root):
        seq, pmap, nodemap = self.make_sequence(root)
        predictions = self.model(self.format_tree(seq))
        for i, node in nodemap.items():
            #node.distribution = predictions[0, i]
            node.distribution = -1 * np.log(predictions[0, i])
