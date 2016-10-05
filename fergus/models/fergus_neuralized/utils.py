from __future__ import print_function
from os.path import join
import random
import baal
from baal.utils import loggers
from baal.structures import DerivationTree
from ikelos.data import Vocabulary, VocabManager
from keras.layers import Embedding, LSTM, Input
from tqdm import tqdm
from collections import Counter
try:
    import cPickle as pickle
except:
    import pickle

def make(node):
    treestr, selfhlf, targethlf, targetgorn = node
    tree = DerivationTree.from_bracketed(treestr)
    tree.set_path_features(self_hlf=selfhlf)
    if targethlf:
        tree.set_path_features(target_hlf=targethlf, target_gorn=targetgorn)
    return tree

def rollout(datum):    
    out = None
    for node in datum:
        dnode  = make(node)

        if any([char in dnode.E.head for char in '1234567890']):
            dnode.E.head = "<NUMBER>"
        if out is None:
            out = dnode
        else:
            out = out.operate(dnode)
            
    return out

def tempfix(s):
    if any([n in s for n in '1234567890']):
        #print(s, "replaced by <NUMBER>")
        #raw_input()
        return "<NUMBER>"
    return s

def create_dataset():
    """
    Target data:
        [(head, spines paths w/ context, children_set), ...]
        s.t. 
        spine_path w/ context will represent the supertag
        a supertag := an elementary tree
        supertag_set := a list of the trees for head word
        children_set := the list of children for that head word
    """
    gist_fp = join(baal.PATH, "science/gist")
    data_fp = join(gist_fp, "data")
    data_files = ["chiang_wsj_train.pkl", "chiang_wsj_dev.pkl", "chiang_wsj_test.pkl"]
    for data_file in tqdm(data_files, desc='data file', position=0):
        with open(join(data_fp, data_file)) as fp:
            data = pickle.load(fp)
        dataset = []
        for datum in tqdm(data, desc="{} data".format(data_file.replace(".pkl","")),
                          position=1):
            #assume path)
            full_tree = rollout(datum)
            roll_features = [(tempfix(a), tempfix(b), c, d, e) for a,b,c,d,e in full_tree.modo_roll_features()]
            dataset.extend(roll_features)
        with open(join(data_fp, data_file.replace(".pkl", "_supertags.pkl")), "w") as fp:
            pickle.dump(dataset, fp)


def load_dataset(filepath):
    with open(filepath) as fp:
        dataset = pickle.load(fp)
    return dataset

def process_data(igor):
    igor.head2spine = {}
    make_filters(igor)
    igor.logger.info("Cutoff Frequency={}.  {} supertags retained; {} supertags cut.".format(igor.data_frequency_cutoff,
                                                                                             len(igor.good_filter),
                                                                                             len(igor.bad_filter)))
    igor.logger.info("")

    igor.train_data, stats = _process_data(igor, igor.train_data)
    igor.logger.info("Training data processed")
    igor.logger.info("{spine_freq} thrown due to frequency; {unseen} thrown due to OOV".format(**stats))
    igor.logger.info("{} data left".format(len(igor.train_data)))


    igor.dev_data, stats = _process_data(igor, igor.dev_data) 
    igor.logger.info("Development/validation data processed")
    igor.logger.info("{spine_freq} thrown due to frequency; {unseen} thrown due to OOV".format(**stats))
    igor.logger.info("{} data left".format(len(igor.dev_data)))

    igor.test_data, stats = _process_data(igor, igor.test_data)
    igor.logger.info("Testing data processed")
    igor.logger.info("{spine_freq} thrown due to frequency; {unseen} thrown due to OOV".format(**stats))
    igor.logger.info("{} data left".format(len(igor.test_data)))

def cartesian_iterator(n_nums):
    import itertools
    duosorted = sorted([(n,i) for i,n in enumerate(n_nums)], reverse=True)
    
    unsort = {orig_idx:new_idx for new_idx, (_, orig_idx) in zip(range(len(n_nums)), duosorted)}
    def f_unsort(indices):
        return [indices[unsort[i]] for i in range(len(n_nums))]

    range_iters = tuple([range(l) for l, _ in duosorted])
    all_idx = sorted(map(f_unsort, list(itertools.product(*range_iters))), key=lambda x: (sum(x), x))
    biggest = max(map(sum, all_idx))
    idx_by_sum = {k:[x for x in all_idx if sum(x)==k] for k in range(1, biggest)}
    for i in range(1,biggest):
        yield idx_by_sum[i]
    #for idx in all_idx:
    #    yield idx
    #iters = [itertools.cycle(xrange(n)) for n in n_nums]
    #indices = [next(it) for it in iters]
    #for i in range(reduce(mul, n_nums)):
    #    indices[i%len(n_nums)] = next(iters[i%len(n_nums)])
    #    yield indices


def compute_parameters(igor):
    igor.max_context_size = max([max([len(n) for n in k]) 
                                 for k in igor.vocabs.spines.keys()])
    igor.max_spine_length = max([len(k) for k in igor.vocabs.spines.keys()])
    igor.max_num_supertags = max(map(len,igor.head2spine.values()))
    igor.spineset_size = igor.max_num_supertags
    igor.max_num_spines = igor.max_num_supertags # preferred vernacular. 
    igor.word_vocab_size = len(igor.vocabs.words)


def _process_data(igor, dataset, build=False):
    out_data = []
    bad_stats = {'spine_freq': 0, 'unseen': 0}    
    keyify = lambda spines: tuple(map(tuple, spines))
    for head, parent_head, bracket, spine, parent_spine in dataset:
        if bracket in igor.bad_filter: 
            bad_stats['spine_freq'] += 1
            continue
        encoded_spine = encode_spine(spine, igor)
        encoded_parent_spine = encode_spine(parent_spine, igor)
        spine_id = igor.vocabs.spines[keyify(encoded_spine)]
        parent_spine_id = igor.vocabs.spines[keyify(encoded_parent_spine)]
        head_id = igor.vocabs.words[head]
        parent_head_id = igor.vocabs.words[parent_head]

        igor.head2spine.setdefault(head_id, set()).add(spine_id)
        igor.head2spine.setdefault(parent_head_id, set()).add(parent_spine_id)

        out_data.append((head_id, parent_head_id, spine_id, parent_spine_id))

    return out_data, bad_stats


def encode_spine(spines, igor):
    encoded_spine = []
    for nodeset in spines:
        encoded_nodeset = []
        for pos, spine_info in nodeset:
            encoded_node = (igor.vocabs.pos.add(pos), 
                            igor.vocabs.type.add(spine_info.upper()))
            encoded_nodeset.append(encoded_node)
        encoded_spine.append(encoded_nodeset)
    return encoded_spine


def make_filters(igor):
    """Filter the data with the data_frequency_cutoff specified in the conf

    - Count on train.  Filter on rest.
    - Since we are filtering based on classification target, remove unseen classes
      in test and dev as well.  
    - Report the number of removed supertags, and the reduction in sizes of train, test, and dev sets
    """
    counts = Counter([bracket for _,_,bracket,_,_ in igor.train_data])
    igor.good_filter = set([ex for ex,count in counts.items() if count>igor.data_frequency_cutoff])
    igor.bad_filter = set([ex for ex,count in counts.items() if count<=igor.data_frequency_cutoff])


if __name__ == "__main__":
    create_dataset()