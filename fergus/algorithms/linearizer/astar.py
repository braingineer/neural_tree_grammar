'''
A* core 

a linearizer takes as input a partial tree
the partial tree is a linked-daughter-list tree with elementary spines assigned to each node
each node can then bind in a limited number of ways given its elementary spine

a* chart parsing
- enumerate position ordering for children
- substitution nodes don't move (bindings could, in theory swap)
- an insertion which has multiple binding spots could change the ordering number of other nodes
- so, for each insertion, for each of their positions, enumerate the gorn orderings
- this is basically finding the permutations based on gorn addressing
- these are assigned indices in order
- children are then put into the *queue with uniqueness enforced on position index
- pubsub configuration used; 
- sub to completed events to find children, open events to find parents
- pub to completed events when finished, open events when still need more children
- algo is:
    + pop from queue
    + publish appropriately
    + subscribe appropriate

'''
from __future__ import absolute_import, print_function, division
import heapq
from collections import defaultdict, deque
from .utils import logprint
from . import utils

import sys
sys.setrecursionlimit(100000)
#sys.settrace()

class StarCache(object):
    def __init__(self):
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
        self.que = deque()
        self.cache = cache._cache 
        self.seen = set()#dict()

    def add(self, state_id):
        score = self.cache[state_id][0]
        self.que.append((score, state_id))
        #heapq.heappush(self.que, (score, state_id))

    def add_many(self, state_ids):
        cache = self.cache
        self.que.extend([(cache[state_id][0], state_id) for state_id in state_ids])
        #self.que = list(heapq.merge(self.que, ((cache[state_id][0], state_id) for state_id in state_ids)))

    def pop(self):
        seen = self.seen
        que = self.que
        score, state_id = que.pop() # heapq.heappop(que)
        while state_id in seen:# and score > seen[state_id]:
            score, state_id = que.pop() #heapq.heappop(que)
        seen.add(state_id)#[state_id] = score
        return state_id
        
    def __len__(self):
        return len(self.que)


class StarEvents(object):
    def __init__(self, cache):
        self.open_events = defaultdict(lambda: set())
        self.completed_events = defaultdict(lambda: set())
        self.cache = cache

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
        subscribe = self.subscribe
        
        state = cache[in_state_id][1]
        #node = self.nodes[state.node_id]
        out_state_ids = []
        if state.completed: ## listeners for complete events get notified. they want daughters
            """ this is a completed event; state is a complete subtree """
            logprint(2, "Publishing to parents")
            listeners = self.completed_events[state.node_id]
            logprint(2, "Parents: ", listeners)
            for listener_id in listeners:
                for new_state in cache[listener_id][1].step(state):
                    # if new_state is None:
                    #     k = "N/A"
                    # else:
                    #     k = new_state.key in self.cache._mapping
                    # logprint(2, "new state (in<{}>): ".format(k), new_state)
                    
                    ### VERY IMPORTANT
                    ## right here some magic is happening
                    ## this is where the uniqueness check takes place
                    ## if the cache deems the state as inferior / redundant, it won't save it
                    ## instead, it'll return the id of an existing state. 
                    ## as a twin move, the subscriptions are on sets, to enforce no duplicates
                    #if new_state is not None:
                    new_state_id = flip(new_state)
                    out_state_ids.append(new_state_id)
                    subscribe(new_state_id, new_state)

        else: ## listeners for open events get fired. they want parents
            """ this is an open event; state is an open subtree and wants to find complete subtrees """
            listeners = self.open_events[state.next_dot]

            logprint(2, "Publishing to daughters")
            logprint(2, "Daughters: ", listeners)

            for listener_id in listeners:
                listener_state = cache[listener_id][1]
                for new_state in state.step(listener_state):
                    # if new_state is None:
                    #     k = "N/A"
                    # else:
                    #     k = new_state.key in self.cache._mapping
                    # logprint(2, "new state (in<{}>): ".format(k), new_state)
                    ## same disclaimer as above
                    #if new_state is not None:
                    new_state_id = flip(new_state)  ## this does the uniqueness check
                    out_state_ids.append(new_state_id)
                    subscribe(new_state_id, new_state)

        
        que.add_many(out_state_ids)
        



def run(partial, verbose=1, return_results=False):
    cache = StarCache()
    eventpipe = StarEvents(cache)
    que = StarQueue(cache)

    states = partial.enumerate_states()
    state_ids = list(map(cache, states))
    for state, state_id in zip(states, state_ids):
        logprint(1, state_id, ":", state)
        eventpipe.subscribe(state_id, state)
    que.add_many(state_ids)
    
    #import pdb
    #pdb.set_trace()
    jic = 0
    q0 = 0
    num_times = 0
    try:
        while que:
            try:
                state_id = que.pop()
            except IndexError as e:
                # queue is now empty
                #eventpipe.run_bad(que)
                #print("que has failed")
                continue
            before = len(que)
            logprint(4, jic, ": ", cache[state_id])
            eventpipe.publish(state_id, que)
            after = len(que)
            if after > 1000 and after > before * 10:
                logprint(1, "explosive growth???")
                #import pdb
                #pdb.set_trace()
            jic += 1
            if not jic % 1000 and verbose:
                print("{}th iteration; Q length: {};".format(jic, len(que)) + 
                      "cache size: {}; last sentence length: {}".format(len(cache._cache), len(cache[state_id].finished)))
    except KeyboardInterrupt as e:
        print("keyboard interrupt")
        import pdb
        pdb.set_trace()

    top_states = eventpipe.open_events[partial.id]
    if utils.level > 0:
        if len(top_states) == 0:
            logprint(1, "EMPTY FINAL STATE")
        else:
            logprint(1, "enumerating {} final states".format(len(top_states)))
            all_strings = list()
            for state_id in top_states:
                state = cache[state_id]
                all_strings.append(format_(sorted(state.finished.items())))
            all_strings = set(all_strings)
            logprint(1, "after unique restriction, {} left".format(len(all_strings)))
            #print("\n".join(all_strings))
            logprint(1, "ending final state enumeration")

    if return_results:
        results = set()
        for state_id in top_states:
            state = cache[state_id]
            results.add(format_(sorted(state.finished.items())))
        return results
        
    #all_ = []
    #for _, state in cache._cache.values():
    #    all_.append([x for x in sorted(state.finished.items())])

    #for state in sorted(all_, key=len, reverse=True):
    #    logprint(3, format_(state))

    #for state in sorted(all_, key=len, reverse=True)[:5]:
    #    logprint(1, format_(state, 0))
    
    #if utils.level >= 2:
    #    import pdb
    #    pdb.set_trace()
    
def format_(item, off=1):
    item = sorted(item)
    l = 0
    s=[]
    for k,v in item:
        #logprint(1+off, k,v)
        if len(k)>l:
            l=len(k)
            #s.append('(')
        elif len(k) < l:
            l = len(k)
            #s.append(')')
        s.append(v)
    #print(s)
    return ' '.join(s)