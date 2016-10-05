"""A dynamic programming algorithm to linearize unordered tree grammar trees

Plan:
    1. each tree orders its children by their gorn prefix
    2. split the children into two groups: left and right
    3. iterate through each prefix group in the left-right groups
    4. going from left to right ala prefix, assign position numbers to children
        - don't worry about situations which offset child indices for no reason
            - this occurs when an insertion can go higher up in the tree and be to the left of others
            - but it also has an option to go to the right of the others. 
            - if we naively enumerate all positions, the possibly-empty position will still be marked
        - assign each children the fully combination number (so if 3 children in group, t+1, t+2, t+3)
    5. once all nodes have their indices, spawn them with it
    6. nodes with no daughters are marked as complete
    7. each other node specifies an numbered iteration over its daughters and looks for hte first
    8. upon finding a daughter, a node accepts it if its position number is not taken
        - this means we will have some duplicate states, but completed daughters could mark their completeness
          to make this not as much of an issue. 

guarantees:
    - because a node doesn't accept any position or child more than once, most states won't repeat

PartialTree's responsibility:
    - enumerate the prefixes and tell all children their possible position ids
    - be able to produce its id, a separate instantiation for each of its position ids, and 
      the list of its daughter ids

State's use of PartialTree:
    - state will ask for ptree to make all of its instantiations
    - it will make a single state for each of them
    - this state will be responsible for tracking the children it's accepted so far
    - the state will accept a child, include its position number in the already done

the only way the numbering system proposed could result in a bad state is if:
    - selecting one child meant that another child's spot was impossible
    - the other spot would also not have been taken yet
    - could we have two substutions at the same site?
    - no, because we don't do the multiple ordering spaces, 
        + nevermind. yes. we need to keep track of the sub valencies so far
        + in addition to the ordering valency
        + because, for example, we don' tknow what comes first, an insertion or a substitution
        + so we allow for both instantiations by giving them two possible child numbers
    - outside of a prefix, the spans are all locked into place. 

better way to talk about things:
    - prefix groups are spans of child ordering indices
    - these spans never overlap or cross
    - inside each span, the ordering of children is up for grabs
    - some spans will end up being empty
    - so the span indices don't match actual indices
    - the span indices are a larger instantiation space on purpose. 

so the partial tree needs to instantiate the prefix groups as spans
then, when called to be made, each individual needs to know their potential prefix groups
the individual should also keep track of the mapping between their ordering number
and the gorn that would be required to do it

though, i guess that's not really needed. for surface form, we only need to know ordering. 
"""

from __future__ import print_function, division, absolute_import
from copy import deepcopy, copy
from baal.structures import DerivationTree
from collections import defaultdict
from .states import StarState, DPState

try:
    range = xrange
except:
    pass


class PartialTree(object):
    id_gen = (i for i in range(10**10))
    def __init__(self, tree, daughters, parent=None, size=0):
        self.tree = tree
        self.head = tree.E.head
        self.daughters = daughters
        self.possible = []
        self.by_gorn = {}
        self.size = size + 1 # for self

        if parent is None:
            PartialTree.id_gen = (i for i in range(10**10))
        self._id = next(PartialTree.id_gen)
        self.parent = parent
        self.idx_set = set()
        self._all_nodes = None
        self.pos2gorn = {}
        self.head_index = None

    @classmethod
    def from_tagged(cls, tagged_tree):
        tree = DerivationTree.from_bracketed(tagged_tree.predicted_tree())
        tree.E.set_path_features(self_hlf=tree.E.hlf_symbol)
        daughters = [PartialTree.from_tagged(d) for d in tagged_tree.daughters]
        size = 0
        for d in daughters:
            d.tree.E.set_path_features(target_hlf=tree.E.hlf_symbol)
            size += d.size
            
        return cls(tree, daughters, size=size)
    
    @classmethod
    def from_list(cls, tree_list):
        tree = DerivationTree.from_bracketed(tree_list[0])
        tree.E.set_path_features(self_hlf=tree.E.hlf_symbol)
        daughters = [PartialTree.from_list(d) for d in tree_list[1]]
        daughters = sorted(daughters, key=lambda d: (d.size, d._id), reverse=True)
        size = 0
        for d in daughters:
            size += d.size
            d.tree.E.set_path_features(target_hlf=tree.E.hlf_symbol)
        return cls(tree, daughters, parent=tree, size=size)
    
    @property
    def all_nodes(self):
        if self._all_nodes is None:
            self._all_nodes = [self]
            for d in self.daughters:
                self._all_nodes.extend(d.all_nodes)
        return self._all_nodes

    def measure_difficulty(self):
        val = max(1, len(self.idx_set))
        for dot in self.daughters:
            val *= dot.measure_difficulty()
        return val
    
    def enumerate_states(self, version='astar'):
        if version=='astar':
            State = StarState
        elif version == 'dp':
            State = DPState
        else:
            raise Exception("invalid state choice")
        nodes = self.all_nodes
        states = []
        only_one = 0
        for node in nodes:
            if len(node.idx_set) == 0:
                node.pos2gorn[0] = (0,)
                states.append(State(node, set([0])))
                only_one += 1
            else:
                states.append(State(node, node.idx_set))
        try:
            assert only_one == 1 #shouldn't happen anymore; filtering in prep
        except:
            import pdb
            pdb.set_trace()
        return states

    def prep(self):
        left_states, right_states = self.annotate_lattice()
        
        idx = 0
        for prefix, group in sorted(left_states.items()):
            idx += self.assign(group, idx)
        self.head_index = idx
        idx += 1
        for prefix, group in sorted(right_states.items()):
            idx += self.assign(group, idx)
        
        self.good_daughters = []
        self.bad_daughters = []
        for daughter in self.daughters:
            if len(daughter.idx_set) > 0:
                self.good_daughters.append(daughter)
                daughter.prep()
            else:
                self.bad_daughters.append(daughter)
                continue
        self.daughters = self.good_daughters

    def annotate_lattice(self):
        left_states, right_states = defaultdict(lambda: []), defaultdict(lambda: [])
        head_addr = self.tree.E.head_address
        points = [pt for _, pt in self.tree.E.point_iterator(True)]
        self.sub_points = set()
        for daughter in self.daughters:
            daughter.idx_set = set()
            for i, point in enumerate(points):
                if point.match(daughter.tree.E.tree_operation):
                    anno_gorn, prefix_gorn, is_ins = self.annotate_gorn(point.gorn, daughter, head_addr)
                    if anno_gorn < head_addr:
                        left_states[prefix_gorn].append((anno_gorn, point.gorn, daughter, is_ins))
                    else:
                        right_states[prefix_gorn].append((anno_gorn, point.gorn, daughter, is_ins))
                    if not is_ins:
                        self.sub_points.add(point.gorn)

        self.sub_points = tuple(self.sub_points)
        return left_states, right_states

    def assign(self, group, base_idx):
        '''assign all possible indices in span to nodes in group

        note:
            insertions and substitutions can go anywhere
            substitutions can't cross other subs. '''

        ### sub bookkeeping
        sub_gorn = [anno_gorn for anno_gorn,_,_,ins_bool in group if not ins_bool]
        sub_gorn_set = set(sub_gorn)
        group_count = len(group) - len(sub_gorn) + len(sub_gorn_set)
        sub_gornmap = {g:i for i,g in enumerate(sorted(sub_gorn_set))}
        num_subgorns = len(sub_gorn_set)

        ### the options ot be assigned
        options = list(range(base_idx, base_idx+group_count))
        ### sort the group by the annotation gorn so we go left to right
        ### this preserves the sub ordering
        group = sorted(group, key=lambda g: g[0])
        #print("Group size: {}; group options: {}".format(len(group), options))
        for i, (anno_gorn, point_gorn, dot_obj, ins_bool) in enumerate(group):
            dot_obj.is_insertion = ins_bool
            if ins_bool:
                # insertions get all of the options
                dot_obj.idx_set.update(options)
            else: 
                # substitutions only get the options that don't cross other subs
                left_post = sub_gornmap[anno_gorn] 
                right_post = -(num_subgorns - (sub_gornmap[anno_gorn] + 1)) or None # coalescing or; defaults to None if left is 0
                # sub slice is number of subs to left (the gornmap index retrieved)
                ## all the way up to the number of gorns to the right. if 0 to the right, then it should be None
                option_slice = slice(left_post, right_post)
                #print("is sub; interval: ({},{}]".format(left_post, right_post))
                option_subset = options[option_slice] # the set of indices w/o crossing
                dot_obj.idx_set.update(option_subset)
                dot_obj.associate(point_gorn, option_subset)
        return group_count

    def annotate_gorn(self, gorn, dot, head_addr):
        direction = dot.tree.E.tree_operation.target['attach_direction']
        if direction == 'right':
            return gorn+(100,), gorn+(100,), True
        elif direction == "left":
            return gorn+(-100,), gorn+(-100,), True
        elif gorn > head_addr:
            return gorn, gorn[:-1]+(100,), False     
        else: 
            return gorn, gorn[:-1]+(-100,), False

    def associate(self, gorn, position_indices):
        self.pos2gorn.update({pidx:gorn for pidx in position_indices})

    def __str__(self):
        return str(self.tree)
    
    def __repr__(self):
        return str(self.tree) + " - " + repr(self.tree)


