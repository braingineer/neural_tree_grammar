from baal.induce import enrich as ten
from .utils import decode_node, decode_spine, encode_spine, cartesian_iterator
import numpy as np

try:
    range = xrange
except:
    pass

class Node(object):
    id_gen = (x for x in range(100000000))
    def __init__(self, node, ref, parent=None):
        self.head = node.head
        if any([n in self.head for n in '1234567890']):
            self.head = "<NUMBER>"  
        self.id = next(Node.id_gen)
        self.ref = ref
        self.parent = parent
        self.head_id = self.ref.igor.vocabs.words[self.head]
        self.head_treelets = []
        self.child_treelet = -1
        self.true_bracket = node.bracketed

        keyify = lambda spines: tuple(map(tuple, spines))
        encoded_spine = encode_spine(node.get_spine(), self.ref.igor)
        self.spine_id = self.ref.igor.vocabs.spines[keyify(encoded_spine)]
        self.selected_tag = None

        self.spineset = [spine_id for spine_id in list(self.ref.igor.get_spineset(self.head_id)) 
                                  if spine_id != self.ref.igor.vocabs.spines.unk_id]
        self.spine_lookup = {s:i for i, s in enumerate(self.spineset)}


    @classmethod
    def from_derivation(cls, tree, ref, parent=None):
        node = cls(tree, ref, parent)
        node.daughters = [cls.from_derivation(child, ref, node) for child in tree.children]
        node.subtree_ids = set([node.id] + [d.id for d in node.daughters])
        node.daughter_lookup = {d.id:i for i, d in enumerate(node.daughters)}
        return node

    @classmethod
    def from_path(cls, path, ref):
        cls.reset()
        tree = ten.rollout(path)
        return cls.from_derivation(tree, ref)
    
    @staticmethod
    def reset():
        Node.id_gen = (x for x in range(1000000))
    
    @property
    def count(self):
        return 1 + sum([d.count for d in self.daughters])
    
    @property
    def treelets(self):
        tlets = [(self, d) for d in self.daughters]
        for d in self.daughters:
            tlets.extend(d.treelets)
        return tlets
    
    @property
    def leaves(self):
        leaves = []
        if len(self.daughters) == 0:
            leaves.append(self)
        else:
            for d in self.daughters:
                leaves.extend(d.leaves)
        return leaves

    @property
    def sequence(self):
        out_seq = []
        if self.parent is None:
            pid = -1
        else:
            pid = self.parent.id
        this = (self, pid, self.head_id, [d.head_id for d in self.daughters])
        out_seq = [this]
        for dot in self.daughters:
            out_seq.extend(dot.sequence)
        return out_seq

    @property
    def flat(self):
        out = {self.id: self}
        for d in self.daughters:
            out.update(d.flat)
        return out

    @property
    def tree(self):
        if self._tree is None:
            self._tree = self.predicted_tree()
        return self._tree

    def initial_score(self, spine_idx):
        raise NotImplementedError

    def compute_best(self):
        raise NotImplementedError

    def compute_downside(self):
        self.daughter_scores = [dot.compute_downside() for dot in self.daughters] 
        self.downside_score = sum(self.daughter_scores) 
        self.compute_best()
        return self.best_score + self.downside_score

    def cache_upside(self, root_score=0):
        '''Cache the negative log likelihood of the best options for the distributions above'''
        self.upside_score = max(0, root_score - self.downside_score - self.best_score)
        for dot in self.daughters:
            dot.cache_upside(root_score) 

    def get_valency_options(self, tag, return_ins=False):
        try:
            sub_pts, ins_pts = self.ref.igor.valency_lookup[tag]
        except KeyError as e:
            return []

        if return_ins:
            return sub_pts, ins_pts
        else:
            return sub_pts

    def to_list(self):
        return self.predicted_tree(), [dot.to_list() for dot in self.daughters]

    def set_supertag(self, spineset_idx):
        self.selected_tag = self.ref.igor.get_spineset(self.head_id)[spineset_idx]

    def predicted_tree(self):
        if self.selected_tag is None:
            return None
        spine = self.ref.igor.vocabs.spines.lookup(self.selected_tag)
        return self.ref.igor.reconstruct(spine, self.head)

    def annotate(self):
        for i, tlet in enumerate(self.treelets):
            tlet[0].head_treelets.append(i)
            assert tlet[1].child_treelet == -1
            tlet[1].child_treelet = i

    def full_comparison_stats(self):
        full_counts = _stats(self)
        for daughter in self.daughters:
            for k,v in daughter.full_comparison_stats().items():
                full_counts[k].extend(v)
        return full_counts

    def __str__(self):
        return self.head+"("+str(self.count)+")"+"->"+"|".join(d.head for d in self.daughters)
    
    def __repr__(self):
        return str(self)


    #def best_score(self):
    #    return 



class FergusNNode(Node):
    def __init__(self, node, ref, parent=None):
        super(FergusNNode, self).__init__(node, ref, parent)
        if parent is not None:
            self.parent_spine_lookup = parent.spine_lookup

        sp_size = self.ref.igor.spineset_size
        if len(node.children) > 0:
            self.leaf = False
            self.distribution = np.zeros((len(node.children), sp_size, sp_size))
        else:
            self.leaf = True
            self.distribution = np.zeros(sp_size)

    def add_distribution(self, daughter_id, dist):
        self.distribution[self.daughter_lookup[daughter_id]] = -1 * np.log(dist)

    def compute_best(self):
        if self.leaf:
            self.best_score = self.distribution.min()
            self.best_index = self.distribution.argmin()
            self.worst_score = self.distribution[:len(self.spineset)].max()
        else:
            self.best_score = self.distribution.min(axis=1).sum(axis=0).min()
            self.best_index = self.distribution.min(axis=1).sum(axis=0).argmin()
            self.worst_score = self.distribution.max(axis=1).sum(axis=0)[:len(self.spineset)].max()

    def initial_score(self, spine_id):
        if self.leaf:
            inside_score = self.distribution[self.spine_lookup[spine_id]]
        else:
            inside_score = self.distribution[:, :, self.spine_lookup[spine_id]].min(axis=1).sum()
        return self.upside_score + inside_score + self.downside_score

    def score_update(self, dot_idx, dot_spine_idx, self_spine_id):
        self_spine_idx = self.spine_lookup[self_spine_id]
        dot_dist = self.distribution[dot_idx, :, self_spine_idx]
        old_inside = dot_dist.min()
        old_downside = self.daughter_scores[dot_idx]
        new_inside = dot_dist[dot_spine_idx]
        return new_inside - old_downside - old_inside

class FergusRNode(Node):
    def compute_best(self):
        self.best_score = self.distribution.min()
        self.best_index = self.distribution.argmin()
        self.worst_score = self.distribution[:len(self.spineset)].max()

    def get_score(self, spine_id):
        return self.distribution[self.spine_lookup[spine_id]]

    def initial_score(self, spine_id):
        return self.upside_score + self.get_score(spine_id) + self.downside_score


def _stats(node):
    '''
    putting down here because it's ugly.
    '''
    ypred = node.ref.igor.vocabs.spines.lookup(node.selected_tag)
    ytrue = node.ref.igor.vocabs.spines.lookup(node.spine_id)
    unk_id = node.ref.igor.vocabs.spines.unk_id
    full_counts = {'correct': [], 'node_syncat': [], 'node_role':[], 'head_syncat': []}
    if node.selected_tag == node.spine_id:
        full_counts['correct'].append(1)
    else:
        full_counts['correct'].append(0)

    if node.selected_tag != unk_id and node.spine_id != unk_id:

        if ypred[0][0][0] == ytrue[0][0][0]:
            full_counts['node_syncat'].append(1)
        else:
            full_counts['node_syncat'].append(0)

        if ypred[0][0][1] == ytrue[0][0][1]:
            full_counts['node_role'].append(1)
        else:
            full_counts['node_role'].append(0)

        if len(ypred) == 1:
            lastpred = ypred
        else:
            lastpred = [n for n in ypred[-1] if node.ref.igor.vocabs.type.lookup(n[1]) == "SPINE"]
            try:
                assert len(lastpred) == 1
            except:
                import pdb
                pdb.set_trace()

        if len(ytrue) == 1:
            lasttrue = ytrue
        else:
            lasttrue = [n for n in ytrue[-1] if node.ref.igor.vocabs.type.lookup(n[1]) == "SPINE"]
            try:
                assert len(lasttrue) == 1
            except:
                import pdb
                pdb.set_trace()

        if lasttrue[0] == lastpred[0]:
            full_counts['head_syncat'].append(1)
        else:
            full_counts['head_syncat'].append(0)

    else:
        ans = full_counts['correct'][-1]
        full_counts['node_syncat'].append(ans)
        full_counts['node_role'].append(ans)
        full_counts['head_syncat'].append(ans)

    return full_counts