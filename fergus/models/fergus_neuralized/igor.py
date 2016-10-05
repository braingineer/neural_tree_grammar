from __future__ import absolute_import, print_function, division

from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import itertools
import os
from os.path import join

from baal.structures import DerivationTree
from keras.layers import Embedding, LSTM, Input
from keras.utils.np_utils import to_categorical
from ikelos.data import VocabManager, Vocabulary, DataServer

try:
    import cPickle as pickle
except:
    import pickle

from . import utils
from ...configs import global_config

class Igor(DataServer):
    """Handle the data for FERGUS 

    Learning Problem 1.
        - Each datapoint has: mother_head, daughter_heads, mother_supertag (e-tree)
        - Given the dataset, we can collect all supertags for a given head
        - Now, each datapoint has: mother_head, possible_supertags, daughter_heads, target_supertag
        - So, form the dataset as:
            X[bth_batch, ith_supertag][:spine_length]=spine_context_matrix
            Y[bth_batch] = to_categorical(truth_index, max_possible_supertags)

        Note: 
            + at run time, we can pre-compute all of the spine
    """ 
    
    def verify_directories(self):
        d1 = os.path.join(self.model_location, self.saving_prefix)
        d = os.path.dirname(d1)
        if not os.path.exists(d):
            os.makedirs(d)
        
    def format_single(self, daughter_head, parent_head, daughter_spine, parent_spine):
        '''Current formulation:
            quadruplet of (daughter head, parent head, daughter spine, parent spine)
                       are observed. there are two ways we can treat the classification
                       so, I'm doing them both. 
            1. classify to correct daughter spine
            2. classify to correct parent spine

            in each case, we need the total spine set for the hypotheses
            we also need the index of the correct one. for simplicity, just going to pass in 2 extra vectors.
        '''
        N_t = self.max_num_spines
        X_dspineset = np.zeros((N_t,))
        X_pspineset = np.zeros((N_t,))
        X_dspine = np.array([daughter_spine])
        X_pspine = np.array([parent_spine])
        X_dhead = np.array([daughter_head])
        X_phead = np.array([parent_head])
        Y_dspine = np.array([-1]) 
        Y_pspine = np.array([-1])

        for i, spine in enumerate(self.head2spine[daughter_head]):
            if spine == daughter_spine:
                assert Y_dspine[0] == -1
                Y_dspine[0] = i
            X_dspineset[i] = spine

        for i, spine in enumerate(self.head2spine[parent_head]):
            if spine == parent_spine:
                assert Y_pspine[0] == -1
                Y_pspine[0] = i
            X_pspineset[i] = spine


        return X_dspineset, X_pspineset, X_dspine, X_pspine, X_dhead, X_phead, Y_dspine, Y_pspine

    def format_batch(self, data):
        B = self.batch_size
        N_t = self.max_num_supertags
        N_s = self.max_spine_length 
        N_c = self.max_context_size
        N_ch = self.max_num_children

        X_dspinesets = np.zeros((B, N_t,), dtype=np.int32)
        X_pspinesets = np.zeros((B, N_t,), dtype=np.int32)
        X_dheads = np.zeros((B, 1), dtype=np.int32)
        X_pheads = np.zeros((B, 1), dtype=np.int32)
        X_dspines = np.zeros((B, 1), dtype=np.int32)
        X_pspines = np.zeros((B, 1), dtype=np.int32)
        Y_dspines = np.zeros((B, N_t), dtype=np.int32)
        Y_pspines = np.zeros((B, N_t), dtype=np.int32)

        for d_i, (X_dset, X_pset, X_ds, X_ps, X_dh, X_ph, Y_ds, Y_ps) in enumerate(data):
            X_dspinesets[d_i] = X_dset
            X_pspinesets[d_i] = X_pset
            X_dheads[d_i] = X_dh
            X_pheads[d_i] = X_ph
            X_dspines[d_i] =  X_ds
            X_pspines[d_i] =  X_ps
            Y_dspines[d_i] = to_categorical(Y_ds, N_t)
            Y_pspines[d_i] = to_categorical(Y_ps, N_t)

        return ([X_dspinesets, X_pspinesets, X_dheads, X_pheads, X_dspines],
                [Y_pspines, Y_dspines])
            
    
    def serve_single(self, data):
        """Serve a supertagger batch

        The supertagger learning is trying to 1) summarize a supertag
                                              2) learn to classify with daughter context
        
        Things to do:
        - For every head, there is usually several elementary trees associated with it
                          This means there are dfiferent spine/contexts. 
        - In the dataset, collect the tagset for every head.
        - Then, the max_num_tagset is the max number any one head will have
                  (i think and had the most)
        - The max spine size is the longest root to head pos
        - The max context size is the large number of children at a spine node

        Find these numbers. A single example is 1 head. 

        This will serve that 1 head. So, 
            1) randomly iterate over all possible heads.
            2) get the tagsets and info for it
            3) yield the resulting matrices 
        """
        for data_i in np.random.choice(len(data), len(data), replace=False):
            head, parent_head, spine, parent_spine = data[data_i]
            if self.vocabs.words.unk_id == head or head not in self.head2spine:  
                continue
            yield self.format_single(head, parent_head, spine, parent_spine)

    
    def serve_batch(self, data):
        dataiter = self.serve_single(data)
        while dataiter:
            next_batch = list(itertools.islice(dataiter, 0, self.batch_size))
            if len(next_batch) < self.batch_size:
                raise StopIteration 
            yield self.format_batch(next_batch)

    def reconstruct(self, spine, head='null'):
        if len(spine) == 1 and len(spine[0])==1:
            ## crappy fix right now. have to figure this out. 
            pos = self.vocabs.pos.lookup(spine[0][0][0])
            typ = self.vocabs.type.lookup(spine[0][0][1])
            assert typ == "UP" or typ == "ROOT"
            return "({} {})".format(pos, head) 

        out = ""
        last = len(spine) - 1
        for i, ctx in enumerate(spine):
            #print(ctx)
            for (pos,typ) in ctx:
                pos = self.vocabs.pos.lookup(pos)
                typ = self.vocabs.type.lookup(typ)
                #print(pos,typ)
                if typ == "LEFT":
                    out += "("+pos+"*"
                elif typ == "RIGHT":
                    out += "(*"+pos
                elif typ == "UP":
                    out += "("+pos
                elif typ == "SUB":
                    out += "("+pos+")"
                elif typ == "SPINE" and i != last:
                    out += "("+pos
                elif typ == "SPINE" and i == last:
                    out += "("+pos+" {})".format(head)
                else:
                    raise Exception("sanity check")
        out += ")"*last
        return out

    def compute_affordances(self):
        ''' important!   THIS IS ABOUT ROWS OPERATING ON COLUMNS

        We want to know what spines can operate on what other spines.
        this matrix encodes that. Since it's not symmetrical, the rows are the op
        and the columns are the operand. 
        '''
        make_legal = True  
        affordance_file = join(self.model_location, self.saving_prefix, 'affordances.npy')
        if os.path.exists(affordance_file):
            self.legal = np.load(affordance_file)
            make_legal = False
            self.logger.info("+ Legal affordances loaded")
        else:
            self.legal = np.zeros((len(self.vocabs.spines), len(self.vocabs.spines)))

        make_legal_root = True
        root_affordance_file = join(self.model_location, self.saving_prefix, 'root_affordances.npy')
        if os.path.exists(root_affordance_file):
            self.legal_root = np.load(root_affordance_file)
            make_legal_root = False
            self.logger.info("+ Legal ROOT affordances loaded")
        else:
            self.legal_root = np.zeros((len(self.vocabs.spines)))

        make_incoming = True
        incoming_valence_file = join(self.model_location, self.saving_prefix, 'incoming_valence.pkl')
        if os.path.exists(incoming_valence_file):
            with open(incoming_valence_file) as fp:
                self.incoming_valence = np.load(incoming_valence_file)
            make_incoming = False
            self.logger.info("+ Incoming valences loaded")
        else:
            self.incoming_valence = dict()

        make_outgoing = True
        outgoing_valence_file = join(self.model_location, self.saving_prefix, 'outgoing_valence.npy')
        if os.path.exists(outgoing_valence_file):
            self.outgoing_valence = np.load(outgoing_valence_file)
            make_outgoing = False
            self.logger.info("+ Outgoing valences loaded")
        else:
            self.outgoing_valence = np.zeros((len(self.vocabs.spines)), dtype=np.int32)

        ### make every time
        self.insertion_ids = set()

        spines = []
        for spine, spine_idx in self.vocabs.spines.items():
            if spine == self.vocabs.spines.unk_symbol  or spine == self.vocabs.spines.mask_symbol:
                continue
            if self.vocabs.type.lookup(spine[0][0][1]) == "ROOT":
                continue

            if self.vocabs.pos.lookup(spine[0][0][0]) == "ROOT" and make_legal_root:
                self.legal_root[spine_idx] = 1            
            spines.append((DerivationTree.from_bracketed(self.reconstruct(spine)), spine_idx))
            if make_incoming:
                point_symbols = [point.pos_symbol for point in spines[-1][0].E.substitution_points]
                point_ids = [self.vocabs.pos[sym] for sym in point_symbols]
                self.incoming_valence[spine_idx] = point_ids

            if make_outgoing:
                self.outgoing_valence[spine_idx] = self.vocabs.pos[spines[-1][0].E.root_pos]

            if spines[-1][0].is_insertion:
                self.insertion_ids.add(spine_idx)



        if make_legal:
            for spine1, spine_idx1 in spines:
                for spine2, spine_idx2 in spines:
                    self.legal[spine_idx1, spine_idx2] = spine2.accepts_op(spine1)
                    self.legal[spine_idx2, spine_idx1] = spine1.accepts_op(spine2)

        if make_legal:
            np.save(affordance_file, self.legal)
            self.logger.info("+ Legal affordances constructed and cached")
        if make_legal_root:
            np.save(root_affordance_file, self.legal_root)
            self.logger.info("+ Legal ROOT affordances constructed and cached")
        if make_incoming:
            with open(incoming_valence_file, 'wb') as fp:
                pickle.dump(self.incoming_valence, fp)
            self.logger.info("+ Incoming valences constructed and cached")
        if make_outgoing:
            np.save(outgoing_valence_file, self.outgoing_valence)
            self.logger.info("+ Outgoing valences constructed and cached")
            self.logger.debug(" ++ Location: {}".format(outgoing_valence_file))


    def consistent_function(self, parent_spine_id):
        parent_valence = self.incoming_valence[parent_spine_id]
        n_args = len(parent_valence)
        def func(daughter_head_ids,  daughter_viterbi_ids):
            dot_valence, spines = self.get_outgoing_valencies(daughter_viterbi_ids, daughter_head_ids)
            subsumes = self._subsume(parent_valence, dot_valence) 
            matches_n = n_args >= len(dot_valence)
            is_legal = all([self.legal[spine_id, parent_spine_id] for spine_id in spines])
            return subsumes and matches_n and is_legal
        return func

    def _subsume(self, lista, listb):
        ''' does a subsume b? '''
        countera, counterb = Counter(lista), Counter(listb)
        for key in countera.keys():
            if countera[key] < counterb[key]:
                return False
        return True

    def get_outgoing_valencies(self, viterbi_indices, head_ids):
        valences = list()
        spines = list()
        for head_id, viterbi_id in zip(head_ids, viterbi_indices):
            spine_set = self.get_spineset(head_id)
            #if viterbi_id >= len(spine_set):
            #    import pdb
            #    pdb.set_trace()
            #else:
            spine_id = spine_set[viterbi_id]
            spines.append(spine_id)
            if spine_id not in self.insertion_ids: 
                valences.append(self.outgoing_valence[spine_id])
        return valences, spines

    def is_insertion(self, viterbi_id, head_id):
        return self.get_spineset(head_id)[viterbi_id] in self.insertion_ids

    def get_spineset(self, head_id):
        try:
            return self.head2spine[head_id]
        except KeyError as e:
            ## most likely, head was not seen because it had a shitty spine..
            return self.head2spine[self.vocabs.words.unk_id]


    def prep(self):

        save_dir = join(self.model_location, self.saving_prefix+'/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        config = global_config()
        data_dir = config['data_dir']
        self.vocman_file = os.path.join(data_dir, self.vocman_file)
        self.vocabs = VocabManager.from_file(self.vocman_file)
        self.vocabs.freeze_all()
        self.total_num_type = len(self.vocabs.type)
        self.total_num_pos = len(self.vocabs.pos)
        self.train_fp = os.path.join(data_dir, self.train_filepath)
        self.dev_fp = os.path.join(data_dir, self.dev_filepath)
        self.test_fp = os.path.join(data_dir, self.test_filepath)
        head2spine_file = os.path.join(data_dir, self.head2spine_file)
    
        self.debugkey = defaultdict(lambda: False)

        if self.force_data_load or self.in_training:
            self.train_data = utils.load_dataset(self.train_fp)
            self.dev_data = utils.load_dataset(self.dev_fp)
            self.test_data = utils.load_dataset(self.test_fp)
            utils.process_data(self)

        if os.path.exists(head2spine_file):
            with open(head2spine_file, 'rb') as fp:
                self.head2spine = pickle.load(fp)
        else:
            self.head2spine = {k:np.array(list(v)) for k,v in self.head2spine.items()}
            with open(head2spine_file, 'wb') as fp:
                pickle.dump(self.head2spine, fp)
        

        utils.compute_parameters(self)
        self.make_spine_tensors()
        if self.embeddings_file and not self.from_checkpoint:
            embeddings_file = os.path.join(data_dir, self.embeddings_file)
            self.saved_embeddings = np.load(embeddings_file)
        else:
            self.saved_embeddings = None

        self.spine_lookup = {}
        self.valency_lookup = {}
        for spine_tuple, key in self.vocabs.spines.items():
            rpos, rtype = spine_tuple[0][0]
            pos = self.vocabs.pos.lookup(rpos)
            typ = "sub" if self.vocabs.type.lookup(rtype)=="UP" else "ins"
            self.spine_lookup[key] = (pos, typ)


            tree = DerivationTree.from_bracketed(self.reconstruct(spine_tuple))
            sub_pts = []
            ins_pts = []
            for pt in tree.E.substitution_points:
                sub_pts.append(pt.pos_symbol)
            for pt in tree.E.insertion_points:
                ins_pts.append(pt.pos_symbol)
            self.valency_lookup[key] = (tuple(sub_pts), tuple(ins_pts))



        
    def make_spine_tensors(self):
        self.spine_lexicon_size = len(self.vocabs.spines)
        self.X_spinepos = np.zeros((self.spine_lexicon_size, 
                               self.max_spine_length, 
                               self.max_context_size), dtype='int32')
        self.X_spinetype = np.zeros((self.spine_lexicon_size, 
                                self.max_spine_length, 
                                self.max_context_size), dtype='int32')
        for spine, index in self.vocabs.spines.iteritems():
            for s, ctx in enumerate(spine):
                for c, node in enumerate(ctx):
                    self.X_spinepos[index,s,c] = node[0]
                    self.X_spinetype[index,s,c] = node[1]