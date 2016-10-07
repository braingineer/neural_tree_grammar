import os
from ikelos.data import Vocabulary 
try:
    import cPickle as pickle
except:
    import pickle

from collections import Counter
import json


def xor(a,b, v=None):
    return (a is not v and b is v) or (a is v and b is not v)

def load_dataset(filepath, add_period=True, max_length=float('inf')):
    f = lambda x: [y.lower() for y in x.replace("\n","").split(" ")] + (["."] if "." not in x[-4:] and add_period else [])
    stats = [0,0]
    out = []
    with open(filepath) as fp:
        if "pkl" in filepath[-4:]:
            dataset = pickle.load(fp)
        elif ".txt" in filepath[-4:]:
            dataset = fp.readlines()
        else:
            raise Exception("unknown dataset file")
        for sent in dataset:
            datum = f(sent)
            if len(datum) <= max_length:
                stats[0] += 1
                out.append(datum)
            else:
                stats[1] += 1
        print("data file: {}. {} kept; {} discarded.".format(filepath, stats[0], stats[1]))
        return out

        # too opaque
        #return filter(lambda x: len(x) < max_length+2, 
        #               [[['<START>']+f(x)+["<END>"] for x in pickle.load(fp)] if len(s) > ])


def vocab(data, frequency_cutoff=None, size_cutoff=None):
    if not xor(frequency_cutoff, size_cutoff):
        raise Exception("one or the other cutoffs please")

    counter = Counter(word for sent in data for word in sent)

    if frequency_cutoff is not None:    
        print("Using a frequency of {} to reduce vocabulary size.".format(frequency_cutoff))
        words = [word for word,count in counter.most_common() if count > frequency_cutoff]
        print("Vocabulary size reduced. {} -> {}".format(len(counter), len(words)))
        
    elif size_cutoff is not None:
        print("Using a cutoff of {} to reduce vocabulary size.".format(size_cutoff))
        words = [word for word,count in counter.most_common(size_cutoff)]
        print("Vocabulary size reduced. {} -> {}".format(len(counter), len(words)))
    
    else:
        raise Exception("should never happen...")
    
    ### MASK HAS TO BE 0 TO BE CORRECTLY FILTERED IN THE MODEL (IT FILTERS 0)
    extras = ["<MASK>", "<UNK>", '<START>', "<END>"]
    return Vocabulary.from_iterable(extras + words)

def wrap(sent):
    return ['<START>'] + sent + ['<END>']

def unk_filter(sent, vocab):
    return [vocab[x] if x in vocab else vocab['<UNK>'] for x in sent]

def num_filter(sent):
    return ['<NUMBER>' if any([n in x for n in '1234567890']) else x for x in sent]

def convert(data, vocab):
    return [unk_filter(num_filter(wrap(sent)),vocab) for sent in data]

def decode(idx_list, vocab, zipped=False):
    if zipped:
        idx_list1 = [x[0] for x in idx_list]
        idx_list2 = [x[1] for x in idx_list]
        return zip(decode(idx_list1, vocab), decode(idx_list2, vocab))
    return list(vocab.lookup_many(idx_list))

def embeddings_exists(igor):
    return os.path.exists(igor.embeddings_file)

def save_embeddings(igor, embeddings, vocab):
    with open(igor.embeddings_file, "w") as fp:
        pickle.dump(embeddings, fp)
    vocab.save(igor.vocab_file)