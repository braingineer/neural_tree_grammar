from __future__ import print_function, division
from baal.structures import DerivationTree
from copy import deepcopy
from collections import OrderedDict, defaultdict
import itertools
import heapq
from copy import copy, deepcopy


def _limit(i, n=500):
    for j in range(100):
        if j**i > n:
            return j-1


def cartesian_iterator(n_nums, lazy=True):
    if lazy:
        n_nums = [len(n) for n in n_nums]
    max_n = _limit(len(n_nums))
    duosorted = sorted([(min(n,max_n),i) for i,n in enumerate(n_nums)], reverse=True)
    
    unsort = {orig_idx:new_idx for new_idx, (_, orig_idx) in zip(xrange(len(n_nums)), duosorted)}
    def f_unsort(indices):
        return [indices[unsort[i]] for i in range(len(n_nums))]

    range_iters = tuple([range(l) for l, _ in duosorted])
    all_idx = sorted(map(f_unsort, list(itertools.product(*range_iters))), key=lambda x: (sum(x), x))
    biggest = max(map(sum, all_idx))
    idx_by_sum = {k:[x for x in all_idx if sum(x)==k] for k in range(1, biggest)}
    for i in range(1,max_n):
        yield idx_by_sum[i]

def decode_node(igor, node):
    p,t = node
    p = igor.vocabs.pos.lookup(p)
    t = igor.vocabs.type.lookup(t)
    return "{}/{}".format(p,t)

def decode_spine(igor, spine_idx=None, spine=None):
    if spine_idx is not None:
        spine = igor.vocabs.spines.lookup(spine_idx)
    elif spine is None:
        raise Exception("crap")

    return "|"+"|-->|".join(["+".join(map(lambda x: decode_node(igor,x), context)) 
                                    for context in spine])+"|"

def encode_spine(spine, igor):
    encoded_spine = []
    for context in spine:
        encoded_nodeset = []
        for pos, spine_info in context:
            encoded_node = (igor.vocabs.pos.add(pos), 
                            igor.vocabs.type.add(spine_info.upper()))
            encoded_nodeset.append(encoded_node)
        encoded_spine.append(encoded_nodeset)
    return encoded_spine

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

class History(object):
    '''maintain a partially linear history
    '''
    def __init__(self, sort_key, initial_tree):
        self.history = OrderedDict()
        self.pmap = {}
        self.traversed = set()
        self.by_xgorn = defaultdict(lambda: [])
        self.sort_key = sort_key
        self._add((0,), (0,), initial_tree)
    
    def add(self, *args):
        clone = deepcopy(self)
        clone._add(*args)
        return clone
        
    def _add(self, xgorn, sgorn, tree):
        parent = tree.E.tree_operation.target['target_hlf']
        treehlf = tree.E.hlf_symbol
        try:
            assert parent is not None or len(self.history) == 0
            if parent is not None and parent in self.history:
                sgorn_ = sgorn
                if tree.is_insertion:
                    sgorn_ = sgorn + tuple([-100 if tree.E.tree_operation.direction=='left' else 100])
                is_left = sgorn_ < self.history[parent]['tree'].E.head_address
                self.pmap[parent].append(treehlf)
            else:
                is_left = False
            self.history[treehlf] = {'tree':tree, 'xgorn':xgorn, 'sgorn':sgorn, 
                                     'index': len(self.history), 
                                     'is_left': is_left}
            self.by_xgorn[xgorn].append(treehlf)
        except Exception as e:
            import pdb
            pdb.set_trace()

        self.traversed.add(treehlf)
        self.pmap[treehlf] = []
    
    def update(self, ref):
        new_tree = ref.compose(self.flatten())
        book = new_tree.expanded_by_hlf()
        for k,v in book.items():
            old_v = self.history[k]['xgorn']
            if old_v != v:
                self.by_xgorn[v] = self.by_xgorn[old_v]
                del self.by_xgorn[old_v]
                self.history[k]['xgorn'] = v
        return new_tree


    def iter_xgorn(self):
        sortable_xgorn = {}
        for k,v in self.by_xgorn.items():
            k_ = tuple([(-100 if k_ < 0 else 100) if isinstance(k_, float) else k_ 
                  for k_ in k])
            if k_ not in sortable_xgorn: 
                sortable_xgorn[k_] = []
            sortable_xgorn[k_].extend(v)
        return [x[1] for x in sorted(sortable_xgorn.items())]

    def flatten(self):
        seen = set()
        out = []
        for treelist in self.iter_xgorn():
            trees = sorted([self.history[hlf] for hlf in treelist], key=self.sort_key)
            for treedict in trees:
                tree = treedict['tree']
                tree.E.set_path_features(target_gorn=treedict['sgorn'])
                out.append(tree)
        return out

        xgorn_grouping = {}
        for treedict in self.history.values():
            tree = treedict['tree']
            hlf = tree.E.hlf_symbol
            if hlf in seen:
                continue
            out.append(tree)
            children = [self.history[child_hlf] for child_hlf in self.pmap[hlf]]
            children = sorted(children, key=self.sort_key)
            for child in children:
                child['tree'].E.set_path_features(target_gorn=child['sgorn'])
            child_hlf = [c['tree'].E.hlf_symbol for c in children]
            seen.update([hlf]+ child_hlf)
            out.extend([c['tree'] for c in children])
        return out


    def __contains__(self, tree):
        if not hasattr(tree, 'E'):
            tree = tree.tree
            assert hasattr(tree, 'E')
        return tree.E.hlf_symbol in self.traversed

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result 

class Sortkeys:
    @staticmethod
    def lefthand(item):
        tree = item['tree']
        mod = -1 if item['is_left'] else 1
        idx = item['index'] + 1 
        return (idx * mod)

    @staticmethod
    def identity(item):
        return item




def expand(state, tg):
    '''
        steps.
            1. get next node to select 
            2. get all possible tags for it consistent with choices so far
            3. add them as search states by getting score , updating selections so far
                - next states are its daughters plus any daughters still left
    '''

    ## nodes we want to search over
    nodes = tg.root.flat
    
    ## state is score, settings so far, things to explore
    score, tag_map, remaining_valencies, toexplore = state
    (child_idx, parent_idx) = toexplore.pop()

    ### things to explore stores two node indices; one is for child to expand
    next_node = nodes[child_idx]
    parent_options = remaining_valencies[parent_idx]
    next_states = []
    #print("have stuff, exploring options")
    ### go thorugh and expand the next node; 
    ### astar_expand(parent_options) expands things tha tonly match the remaining valency 
    ### the parent_options is just a list of substitution parts of speech
    best = {}
    for new_score, super_tag, tag_flag in next_node.astar_expand(parent_options):
        #print("exploring new options")

        new_rv = deepcopy(remaining_valencies)
        if tag_flag:
            ## tag_flag is None for insertion trees; root pos otherwise
            new_rv[parent_idx].remove(tag_flag)
        ## update the valencies with the things this node accepts
        new_rv[child_idx] = next_node.get_valency_options(super_tag)

        

        newtag_map = deepcopy(tag_map)
        newtag_map[child_idx] = super_tag


        key = scorekey(nodes, new_rv, newtag_map)
        if key in tg.seen:
            if new_score < best[key]:
                best[key] = new_score
            else:
                continue
        tg.seen.add(key)
        best[key] = new_score

        

        ### inner score. sum of tag scores so far
        new_score = sum([nodes[idx].get_score(tag) for idx,tag in newtag_map.items()
                                                   if idx != -1])

        ### outter score. sum of best tags for everything else. 
        other_nodes = [node_id for node_id, node in nodes.items() if node_id not in newtag_map.keys()]
        new_score += sum([nodes[idx].best_score() for idx in other_nodes])

        ### put this in our tag map; this is our eventual goal


        newtoexplore =  [(d.id, child_idx) for d in next_node.daughters] + copy(toexplore)

        ### make the new valency constraints


        #next_states[key] = (new_score, newtag_map, new_rv, newtoexplore)
        next_states.append((new_score, newtag_map, new_rv, newtoexplore))

    #import pdb
    #pdb.set_trace()
    return next_states#.values()

def scorekey(nodes, state_var, state_var2):
    nodes = sorted(nodes.items())
    scorekey = tuple([((state_var2[node_id], tuple(state_var[node_id])) if node_id in state_var.keys() else None) for node_id, node in nodes])
    return scorekey


def run_astar(tg, return_on_solution=False, verbose=0):
    '''
        run a* algorithm on fergus-r
        state: (score computed from inside & outside scores, 
                mapping from node ids to selected tags, 
                mapping from node ids to remaining valencies,
                node ids for things that can be explored (b/c parent has tag))
    '''
    state = (0., {-1:'ROOT'}, {-1:['ROOT']}, [(tg.root.id, -1)])
    stk = [state]
    solutions = []
    tg.seen = set()
    import time
    start = time.time()
    #print('starting')
    t = 0
    while stk:

        #print('popping off heap;')
        next_state = heapq.heappop(stk)

        t += 1
        if t % 10 == 0:
            #print(len(stk), next_state[0])
            if time.time() - start > 15:
                raise Exception  

        if len(next_state[-1]) == 0:
            if -1 in next_state[1]:
                next_state[1].pop(-1)
            if verbose:
                print("found solution")
            if return_on_solution:
                return next_state
            else:
                solutions.append(next_state)

        #print('expanding')
        new_states = expand(next_state, tg)
        if len(new_states) > 0:
            stk = list(heapq.merge(stk, new_states))


    return solutions