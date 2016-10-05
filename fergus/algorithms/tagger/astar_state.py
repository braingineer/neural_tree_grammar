from copy import copy, deepcopy

class BaseState(object):
    def __init__(self, node, spine_id):
        self.node_id = node.id
        self.spine_id = spine_id
        try:
            assert spine_id > 1
        except:
            import pdb
            pdb.set_trace()
        #tree_str = node.ref.make_tree(spine_id, 'foo')
        self.op_pos, self.op_type = node.ref.igor.spine_lookup[spine_id]
        #tree_op = node.hypothetical_tree(spine_id).E.tree_operation
        #self.op_pos = tree_op.target['pos_symbol']
        #self.op_type = 'ins' if tree_op.is_insertion else 'sub'
        
        self.daughter_ids = tuple(d.id for d in node.daughters)
        self.finished_dots = tuple()
        self.n_finished = 0
        self.n_daughters = len(self.daughter_ids)
        if len(self.daughter_ids) > 0:
            self.next_dot = self.daughter_ids[0]        
        else:
            self.next_dot = None
        self.sub_valency, self.ins_valency = node.get_valency_options(spine_id, True)

        self.score = node.initial_score(spine_id)
        self.daughter_scores = node.daughter_scores
        self.rep_string = None
        #self.rep_string = self.make_str()
        self.key = hash((self.node_id, self.op_type, self.op_pos, self.ins_valency, 
                         self.n_finished, self.next_dot, self.sub_valency))
    @classmethod
    def make_all(cls, node, tg):
        #spineset = node.spineset#tg.igor.get_spineset(node.head_id)
        return [cls(node, spine) for spine in node.spineset if spine != tg.igor.vocabs.spines.unk_id]

    ### IMPORTANT. Implement this.
    def step(self, state_id, state, node):
        ''' returns deep copy of self that's taken one step daughter id'''
        ## early failure
        state_op_type = state.op_type
        state_op_pos = state.op_pos
        sub_valency = self.sub_valency
        
        if not state.completed:
            return None
        if state_op_type == 'sub' and state_op_pos not in sub_valency:
            return None
        if state_op_type == 'ins' and state_op_pos not in self.ins_valency:
            return None       
            
        daughter_id = state.node_id
        spine_id = state.spine_id
        assert daughter_id == self.next_dot
        cls = self.__class__
        new_state = cls.__new__(cls)
        new_state.__dict__.update(self.__dict__)
        assert new_state.spine_id > 1

        new_state.finished_dots = new_state.finished_dots + (state_id,)
        new_state.n_finished += 1
        if new_state.n_finished < new_state.n_daughters:
            new_state.next_dot = new_state.daughter_ids[new_state.n_finished]
        else:
            new_state.next_dot = None
            
        if state_op_type == 'sub':
            assert len(sub_valency) > 0
            del_idx = sub_valency.index(state_op_pos)
            new_state.sub_valency = sub_valency[:del_idx] + sub_valency[del_idx+1:]

        ### remove the outside score associatd with this child 
        new_state.score += self.update_score(spine_id, node)
        
        ### add in the child's inside score (self score + downside score)
        new_state.score += state.score - node.upside_score
        new_state.key = hash((self.node_id, self.op_type, self.op_pos, self.ins_valency, 
                              new_state.n_finished, new_state.next_dot, new_state.sub_valency))
        return new_state
    ###

    def step_bad(self, node_id, score_offset):
        cls = self.__class__
        new_state = cls.__new__(cls)
        new_state.__dict__.update(self.__dict__)
        new_state.finished_dots = new_state.finished_dots + (None, )
        new_state.n_finished += 1
        if new_state.n_finished < new_state.n_daughters:
            new_state.next_dot = new_state.daughter_ids[new_state.n_finished]
        else:
            new_state.next_dot = None
        new_state.score += score_offset 

        new_state.key = hash((self.node_id, self.op_type, self.op_pos, self.ins_valency, 
                              new_state.n_finished, new_state.next_dot, new_state.sub_valency))
        return new_state


    @property
    def completed(self):
        return self.next_dot is None

    def result_map(self, cache):
        if not self.completed:
            print("Node not done!")
        out = {self.node_id:self.spine_id}
        for daughter_id in self.finished_dots:
            if daughter_id is None: continue
            daughter = cache[daughter_id]
            out.update(daughter.result_map(cache))
        return out

    def make_key(self):
        self.key = hash((self.node_id, self.op_type, self.op_pos, self.n_finished, 
                         self.next_dot, self.sub_valency, self.ins_valency))

    def __hash__(self):
        return self.key

    def make_str(self):
        out = "node:{}--op:({},{})".format(self.node_id, self.op_type, self.op_pos)
        ndone, nleft = len(self.finished_dots), len(self.daughter_ids) - len(self.finished_dots)
        out += "--dots_finished:{}--dots_left:{}--next_dot:{}".format(ndone, nleft, self.next_dot)
        out += "--open_sub_valency:{}--open_ins_valency:{}".format(str(self.sub_valency), 
                                                                  str(self.ins_valency)) 
        return out

    def __str__(self):
        if self.rep_string is None:
            self.rep_string = self.make_str()
        return self.rep_string

    def __repr__(self):
        out = "node: {} -- op: ({},{})".format(self.node_id, self.op_type, self.op_pos)
        return out

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if isinstance(k, list):
                setattr(result, k, copy(v))
            else:
                setattr(result, k, v)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result        



class RState(BaseState):    
    def update_score(self, *args):
        return -self.daughter_scores[self.n_finished]

class ZeroState(BaseState):
    def update_score(self, spine_id, node):     
        return node.parent.score_update(self.n_finished, node.spine_lookup[spine_id], self.spine_id)

