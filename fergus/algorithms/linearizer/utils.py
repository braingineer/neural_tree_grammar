from __future__ import print_function 
import psutil
from collections import defaultdict
import numpy as np
from copy import copy
import ikelos as ike

level = 0
## for now, i'm lazy. everything debugs. 
logger = ike.utils.loggers.duallog('linearizer')
tolog = {i:logger.debug for i in range(1, 6)}
tolog[0] = logger.info

try:
    range = xrange
except:
    pass

def logprint(lvl, *msg): 
    # return
    #print(lvl, level)
    if level >= lvl:
        tolog[lvl](' '.join(map(str, msg)))
        #print(*msg)
        

class MemoryException(Exception):
    def __init__(self, msg):
        self.msg = msg
        self.usage = psutil.virtual_memory().percent
    def __str__(self):
        return "OutOfMemory<{}%%>: {}".format(self.usage, self.msg)

    

def decode_one(memos, seq_idx, verbose=0):
    bwd_memo, fwd_memo = memos
    seq_idx, item = bwd_memo[seq_idx]
    out = [item]
    while seq_idx >= 0:
        if verbose: print(out)
        seq_idx, item = bwd_memo[seq_idx]        
        out.append(item)
        if verbose: 
            print(seq_idx)
            print(out)
            print("--")
    return out[::-1]
    
def frontier(memos, seq_idx):
    bwd_memo, fwd_memo = memos
    return fwd_memo[seq_idx]
    
def prefix_decoding(root_state, state_map):    
    idx_iter = (i for i in range(10**10))
    bwd_memo = defaultdict(lambda: next(idx_iter))
    fwd_memo = defaultdict(lambda: set())
    memos = (bwd_memo, fwd_memo)
    final_sequences = _prefix_decoding(root_state, state_map, memos)
    return final_sequences, memos
    
def prefix_step(memos, in_idx, item):
    bwd_memo, fwd_memo = memos
    out_idx = bwd_memo[in_idx, item]
    bwd_memo[out_idx] = (in_idx, item)
    fwd_memo[in_idx].add(out_idx)
    return out_idx
    
def _prefix_decoding(state, state_map, memos, in_seqs=None):
    bwd_memo, fwd_memo = memos
    #print(in_seqs)
    #if in_seqs is not None and ('S'+str(state.id), tuple(in_seqs)) in bwd_memo:
    #    return bwd_memo['S'+str(state.id), tuple(in_seqs)]
    all_seqs = []
    for inner_state in state.inner_states:
        seqs = in_seqs or [-1]
        for pos_idx, item in sorted(inner_state.items()):
            if isinstance(item, int):
                try:
                    assert isinstance(seqs, list)
                except:
                    import pdb
                    pdb.set_trace()
                seqs = _prefix_decoding(state_map[item], state_map, memos, seqs)                
            else:
                seqs = [prefix_step(memos, seq_state, item) for seq_state in seqs]
        all_seqs.extend(seqs)
    #if in_seqs is not None:
    #    bwd_memo['S'+str(state.id), tuple(in_seqs)] = all_seqs
    return all_seqs
        
def decoding(state, state_map, memo=None):
    '''
    base case: all decoding(state_map[item]) return the string case inside the loop
               and, seqs is empty.
               so, it enumerates different ways of ordering strings
               each ordering is a different inner state
                 
    '''
    #if psutil.virtual_memory().percent > 90.:
    #    raise MemoryException("decoding explosion")
    memo = memo or {}
    if state.id in memo:
        return memo[state.id]
    all_s = []
    for inner_state in state.inner_states:
        seqs = [tuple()]
        for _, item in sorted(inner_state.items()):
            if isinstance(item, int):
                new_seqs = []
                for subseq in decoding(state_map[item], state_map, memo):
                    new_seqs.extend([seq + subseq for seq in seqs])
                seqs = new_seqs
            else: # item is string; add it to the sequences so far
                # if everything in this state is a string, it's a simple enumeration  
                # and there is only one seq
                assert isinstance(item, str)
                seqs = [seq+(item,) for seq in seqs]
        all_s.extend(seqs)
    memo[state.id] = all_s
    return all_s
        
        

def last(amap, i, j):
    from copy import copy
    if (i,j) in amap:
        return copy(amap[i,j])
    else:
        return []


def gendist(s0, t0, decode=False):
    if isinstance(s0, str):
        s0 = s0.split(" ")
    if isinstance(t0, str):
        t0 = t0.split(" ")
    vocab = {c:i for i,c in enumerate(set(s0+t0))}
    f_v = lambda x: vocab[x]
    lookup = {i:c for c,i in vocab.items()}
    f_l = lambda x: lookup[x]

    s = list(map(f_v, s0))
    t = list(map(f_v, t0))

    delmap = {(i,0): list(s[:i]) for i in range(len(s)+1)} ## deleting across first column
    insmap = {(0,j): list(t[:j]) for j in range(len(t)+1)} ## inserting across first row

    m = np.zeros((len(s)+1, len(t)+1))
    m[:,0] = np.arange(len(s)+1)
    m[0,:] = np.arange(len(t)+1)

    backpointer = dict()

    ### deleting is moving down the row without moving the column
    ### inserting is moving across the column without moving the row
    #print(vocab)
    for y,row_i in enumerate(s, 1):
        for x, col_j in enumerate(t, 1):
            #print(x,y)
            #print(m)
            #print('--------')
            subcost = m[y-1, x-1] + (row_i != col_j)
            delcost = m[y-1, x] + (row_i not in insmap[y-1,x])
            inscost = m[y, x-1] + (col_j not in delmap[y, x-1])

            if subcost < delcost and subcost < inscost:
                #print("subcost is min with ", subcost, y-1, x-1)
                m[y,x] = subcost
                insmap[y,x] = last(insmap, y-1, x-1)
                delmap[y,x] = last(delmap, y-1, x-1)
                if row_i == col_j:
                    backpointer[y,x] = ('match', y-1, x-1)
                else:
                    backpointer[y,x] = ('sub', y-1, x-1)
            elif delcost <= inscost:
               #print("delcost is min with ", delcost, y-1, x, insmap[y-1, x], f_l(row_i))
                m[y,x] = delcost
                insmap[y,x] = last(insmap, y-1, x)
                delmap[y,x] = last(delmap, y-1, x)
                if row_i in insmap[y, x]:
                    backpointer[y,x] = ('move', y-1, x)
                    insmap[y,x].remove(row_i)
                else:
                    backpointer[y,x] = ('del', y-1, x)
                    delmap[y,x].append(row_i)
            else:
                #print("inscost is min with", inscost, y, x-1, delmap[y, x-1], f_l(col_j))
                m[y,x] = inscost
                insmap[y,x] = last(insmap, y, x-1)
                delmap[y,x] = last(delmap, y, x-1)

                if col_j in delmap[y, x]:
                    backpointer[y,x] = ('move', y, x-1)
                    delmap[y,x].remove(col_j)
                else:
                    backpointer[y,x] = ('ins', y, x-1)
                    insmap[y,x].append(col_j)
    if decode:
        F = lambda wordi, wordj, step: 'from %s to %s by %s' % (wordi, wordj, step)
        wi = lambda y: f_l(s[y])+"(%d)"%y
        wj = lambda x: f_l(t[x])+"(%d)"%x
        
        y,x = len(s), len(t)
        solution, backy, backx = backpointer[y,x]
        steps = [F(wi(y-1), wj(x-1), solution)]
        
        while (backy,backx) in backpointer:
            solution, backy1, backx1 = backpointer[backy, backx]
            steps.append(F(wi(backy-1), wj(backx-1), solution))
            backy = backy1
            backx = backx1
        
        
        print("-")
        print(" ".join(s0))
        print(" ".join(t0))
        print("-")
        print("\n".join(steps[::-1]))
        
    return m[-1, -1]
