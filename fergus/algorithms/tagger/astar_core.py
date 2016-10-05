'''
run a* chart decoder for supertag classification

procedure:
    - pop highest priority
    - if complete, find parent set
    - else, find next daughter set
    - combine with the set
    - add results back to the priority queue

priority:
    - sum of inside-outside values
    - similar to mike lewis's priority 
        + if not exactly it
'''
from __future__ import absolute_import, print_function, division
import pickle
import baal.induce.enrich as ten
from gist import PATH#, fergus
#import fergus.algorithms.tree_chooser as tc
#import fergus.algorithms.tree_chooser_r as tcr
#from fergus.algorithms import linearizer, driver
from copy import copy, deepcopy
import numpy as np
import heapq
from collections import defaultdict



class StarCache(object):
    def __init__(self, tg_hook):
        self._cache = dict()
        self._mapping = dict()
        self._flip = dict()
        self._id = 0
    
    def __getitem__(self, state_id):
        return self._cache[state_id][1]
            
    def __call__(self, state):
        _mapping = self._mapping
        _flip = self._flip
        _cache = self._cache
        
        state_key = state.key
        score = state.score
        
        if state_key in _mapping:
            # want the min score
            state_id = _mapping[state_key] 
            if _cache[state_id][0] >= score:
                _cache[state_id] = (score, state)
            return state_id
        else:
            state_id = self._id
            _cache[state_id] = (score, state)
            _mapping[state_key] = state_id
            _flip[state_id] = state_key
            self._id += 1
            return state_id
            
    def solution(self, states=None, state_ids=None, all_solutions=False):
        states = states or [self[state_id] for state_id in state_ids]
        states_ = filter(lambda state: state.completed and state.op_pos=='ROOT', 
                        states)
        if len(states_) == 0:
            states_ = filter(lambda state: state.op_pos=='ROOT', states)
            if len(states_) == 0:
                return False
                import pdb
                pdb.set_trace()
                print("something wrong with states.. it's 0-length")
        if all_solutions:
            return sorted(states_, key=lambda state: state.score)
        else:
            return min(states_, key=lambda state: state.score)
    
    def __len__(self):
        return len(self._cache)

class StarQueue(object):
    def __init__(self, cache):
        self.que = []
        self.cache = cache._cache 
        self.seen = dict()

    def add(self, state_id):
        score = self.cache[state_id][0]
        heapq.heappush(self.que, (score, state_id))

    def add_many(self, state_ids):
        cache = self.cache
        self.que = list(heapq.merge(self.que, ((cache[state_id][0], state_id) for state_id in state_ids)))

    def pop(self):
        seen = self.seen
        que = self.que
        score, state_id = heapq.heappop(que)
        while state_id in seen:# and score > seen[state_id]:
            score, state_id = heapq.heappop(que)
        seen[state_id] = score
        return state_id
        
    def __len__(self):
        return len(self.que)


class StarEvents(object):
    def __init__(self, cache, tg):
        self.open_events = defaultdict(lambda: set())
        self.completed_events = defaultdict(lambda: set())
        self.cache = cache
        self.tg = tg
        self.root_solution_ticker = 0
        self.possible_bad = {}

    def subscribe(self, state_id, state=None):
        state = state or self.cache[state_id]
        if state.completed:  # if it's complete, it wants parents, so it listens for open
            self.open_events[state.node_id].add(state_id)
        else: # if it's open, it wants daughters and listens for completed subtrees 
            self.completed_events[state.next_dot].add(state_id)

    def subscribe_many(self, state_ids):
        for state_id in state_ids:
            self.subscribe(state_id)

    def publish(self, in_state_id, que):
        cache = self.cache._cache
        flip = self.cache
        nodes = self.tg.nodes
        subscribe = self.subscribe
        
        state = cache[in_state_id][1]
        node = nodes[state.node_id]
        out_state_ids = []
        if state.completed: ## listeners for complete events get notified. they want daughters
            listeners = self.completed_events[state.node_id]
            for listener_id in listeners:
                new_state = cache[listener_id][1].step(in_state_id, state, node)
                ### VERY IMPORTANT
                ## right here some magic is happening
                ## this is where the uniqueness check takes place
                ## if the cache deems the state as inferior / redundant, it won't save it
                ## instead, it'll return the id of an existing state. 
                ## as a twin move, the subscriptions are on sets, to enforce no duplicates
                if new_state is not None:
                    new_state_id = flip(new_state)
                    out_state_ids.append(new_state_id)
                    subscribe(new_state_id, new_state)
        else: ## listeners for open events get fired. they want parents
            listeners = self.open_events[state.next_dot]
            for listener_id in listeners:
                listener_state = cache[listener_id][1]
                listener_node = nodes[listener_state.node_id]
                new_state = state.step(listener_id, listener_state, listener_node)
                ## same disclaimer as above
                if new_state is not None:
                    new_state_id = flip(new_state)  ## this 
                    out_state_ids.append(new_state_id)
                    subscribe(new_state_id, new_state)

        if len(out_state_ids) == 0 and state.completed:
            self.possible_bad[state.node_id] = node.worst_score - node.best_score
        else:
            que.add_many(out_state_ids)
        
        if len(self.open_events[self.tg.root.id]) > self.root_solution_ticker:
            self.root_solution_ticker = len(self.open_events[self.tg.root.id])
            return True
        else:
            return False

    def run_bad(self, que):
        print('desparate')
        cache = self.cache._cache
        flip = self.cache
        subscribe = self.subscribe
        new_ids = []
        for node_id, score_offset in self.possible_bad.items():
            for listener_id in self.completed_events[node_id]:
                new_state = cache[listener_id][1].step_bad(node_id, score_offset)
                state_id = flip(new_state)
                subscribe(state_id, new_state)
                new_ids.append(state_id)
        self.possible_bad = {}
        que.add_many(new_ids)



def decode_solution(tg, cache, root):
    if root:
        for node_id, spine_id in root.result_map(cache).items():
            tg.nodes[node_id].selected_tag = spine_id
    
    x=0
    for node in tg.nodes.values(): 
        if node.selected_tag is None:
            x+=1
            node.selected_tag = node.spineset[node.best_index]
            #import pdb
            #pdb.set_trace()
    if x>0:
        print(x/float(len(tg.nodes))*100, " percent manually set...")

def run(tg, verbose=0):
    cache = StarCache(tg)
    eventpipe = StarEvents(cache, tg)
    que = StarQueue(cache)
    
    for node in tg.nodes.values():
        states = tg.model.State.make_all(node, tg)
        state_ids = map(cache, states)
        if len(state_ids) > 0:    
            eventpipe.subscribe_many(state_ids)
            que.add_many(state_ids)

    #import pdb
    #pdb.set_trace()
    jic = 0
    
    while que:
        try:
            state_id = que.pop()
        except IndexError as e:
            # queue is now empty
            eventpipe.run_bad(que)
            continue
        if eventpipe.publish(state_id, que):
            states = cache.solution(state_ids=eventpipe.open_events[tg.root.id])
            if states:
                decode_solution(tg, cache, states)
                return
                print("Found solution; ", tg.stats()['correct'], 'correct')
        jic += 1
        if not jic % 10 and verbose:
            print("{}th iteration; Q length: {}".format(jic, len(que)))
    
    root_id = tg.root.id
    if len(eventpipe.open_events[root_id]) == 0:
        root_states = [state for _, state in cache._cache.values() if state.node_id == root_id]
        solution = cache.solution(states=root_states)
    else:
        root_state_ids = eventpipe.open_events[root_id]
        solution = cache.solution(state_ids=root_state_ids)
    
    decode_solution(tg, cache, solution)