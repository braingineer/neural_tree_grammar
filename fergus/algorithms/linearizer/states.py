from copy import deepcopy 
from . import utils

class StarState(object):
    def __init__(self, node, position_indices): 
        self.node_id = node._id
        #self.sub_gorn = None if node.tree.is_insertion else node.pos2gorn[position_index]
        self.is_insertion = node.tree.is_insertion
        self.sub_points = node.sub_points
        self.pos2gorn = node.pos2gorn

        self.dot_ids = [d._id for d in node.daughters]
        self.dot_map = {d._id:d.head for d in node.daughters}
        self.n_daughters = len(self.dot_map)
        self.next_dot = (self.dot_ids[0] if (len(self.dot_ids)>0) else None)

        self.finished = {(0,node.head_index,):node.head}

        self.positions_filled = tuple()

        self.n_finished = 0
        self.position_indices = position_indices
        if self.completed:
            self.seq = tuple(x[1] for x in sorted(self.finished.items()))
        else:
            self.seq = tuple(sorted(self.finished.items()))
        # self.key = hash((self.node_id, self.position_index, 
        #                  self.n_finished, 
        #                  self.sub_points,
        #                  self.positions_filled,
        #                  self.seq))
        self.key = hash((self.node_id, self.seq))

        self.score = -1


    def step(self, child_state):        
        cls = self.__class__
        for pos_idx in child_state.position_indices:
            if pos_idx in self.positions_filled:
                continue
                #return None

            if not child_state.is_insertion:
                gorn = child_state.pos2gorn[pos_idx]
                if gorn not in self.sub_points:
                    #return None
                    continue
                else:
                    new_subs = tuple([pt for pt in self.sub_points if pt != gorn]) 
            else:
                new_subs = self.sub_points

            try:
                assert child_state.node_id == self.next_dot
            except:
                import pdb
                pdb.set_trace()

            new_state = cls.__new__(cls)
            new_state.__dict__.update(self.__dict__)

            new_state.sub_points = new_subs

            new_state.finished = copy(self.finished)
            new_state.finished.update({(0,pos_idx,)+k[1:]:v 
                                       for k,v in child_state.finished.items()})

            try:
                assert len(new_state.finished) == len(self.finished) + len(child_state.finished)
            except:
                import pdb
                pdb.set_trace()
            #new_state.finished[child_state.position_index] = self.dot_map[child_state.node_id]
            new_state.n_finished += 1
            if new_state.n_finished < new_state.n_daughters:
                new_state.next_dot = new_state.dot_ids[new_state.n_finished]
            else:
                new_state.next_dot = None

            if new_state.completed:
                new_state.seq = tuple(x[1] for x in sorted(new_state.finished.items()))
            else:
                new_state.seq = tuple(sorted(new_state.finished.items()))

            new_state.score += child_state.score
            new_state.positions_filled = self.positions_filled + (pos_idx,)

            # new_state.key = hash((new_state.node_id, new_state.position_index, 
            #                       new_state.n_finished, 
            #                       new_state.sub_points,
            #                       new_state.positions_filled,
            #                       new_state.seq))
            new_state.key = hash((new_state.node_id, new_state.seq))
            yield new_state

    @property
    def completed(self):
        return self.next_dot is None

    def __hash__(self):
        return self.key

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "node<{}>---seq<{}>".format(self.node_id, self.seq)
        
        
        
class DPState:
    def __init__(self, node, position_indices):
        self.inner_states = [{node.head_index: node.head}]
                              
        self.dot_ids = [d._id for d in node.daughters]
        self.dot_map = {d._id:d.head for d in node.daughters}
        self.n_daughters = len(self.dot_map)
        self.next_dot = (self.dot_ids[0] if (len(self.dot_ids)>0) else None)
        self.n_finished = 0
        self.advanced = False
        self._id = node._id
        self.position_indices = position_indices
        
    #     self.inner_state_count = 1
    #     self.inner_state_set = set([(0, node.head_index)])
    #     self.inner_state_dict = {(0,node.head_index): node.head}
        
    # def _step(self, child):
    #     stepped_states = []
    #     utils.logrpint(1, 'stepv2 start')
    #     state_s = self.inner_state_set
    #     state_d = self.inner_state_dict
    #     child_id = child._id
    #     for pos_idx in child.position_indices:
    #         for state_idx in range(self.inner_state_count):
    #             new_k = (state_idx, pos_idx)
    #             if new_k in state_s:
    #                 continue
    #             state_s.add(new_k)
    #             state_d[new_k] = child_id
        
    def step(self, child):
        '''step through the child's possible position indices and add it
           to any inner state that doesn't have that index already filled
       
        this formulation keeps the child's combinatorics methodologically private (inside inner states)
        this way, we deal with the enumeartions in the decoding and not the search
        '''
        # keep track of the new inner states
        stepped_states = []
        # go through all the places the child could be place
        utils.logprint(1, "going through the child indices in step")
        child_id = child._id
        for pos_idx in child.position_indices:
            # go through all of the inner states for this state
            utils.logprint(1, "iterating over inner states.. stepped<{}>, inner<{}>".format(len(stepped_states), len(self.inner_states)))
            for inner_state in self.inner_states:
                # if its already filled, this inner state is potentially dropped
                # it's dropped because if it doesn't advance a dot, it serves no use
                if pos_idx not in inner_state:
                    stepped_state = {pos_idx:child_id}
                    for k, v in inner_state.items():
                        stepped_state[k] = v
                    stepped_states.append(stepped_state)

                
                #stepped_state = deepcopy(inner_state)
                # newset = set(x for x in inner_state['filled'])
                # newdict = {k:v for k,v in inner_state['contents'].items()}
                # stepped_state = {}
                # stepped_state['filled'] = newset
                # stepped_state['contents'] = newdict
                # stepped_state['filled'].add(pos_idx)
                # # add the child as mapped from the position index
                # # later, this lets us enumerate the children and recurse on their combinatorics
                # stepped_state['contents'][pos_idx] = child._id 
                # stepped_states.append(stepped_state)
        # for sanity: anticipate children with no positions can be combined. indicates failure @ supertag level
        # try:
        #     assert len(stepped_states) > 0
        # except:
        #     raise Exception()
        #     print("node didn't advance any dots; this implies error; investigate")
        #     import pdb
        #     pdb.set_trace()
        if len(stepped_states) > 0:
            del self.inner_states
            self.inner_states = stepped_states
        else:
            utils.logprint(1, "Failure of dot advancement")
        
        self.n_finished += 1
        if self.n_finished < self.n_daughters:
            self.next_dot = self.dot_ids[self.n_finished]
        else:
            self.next_dot = None
            
    @property
    def completed(self):
        return self.next_dot is None

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "node<{}>---|inner_states|={}".format(self._id, len(self.inner_states))