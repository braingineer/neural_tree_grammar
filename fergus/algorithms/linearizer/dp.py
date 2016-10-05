'''run the dynamic program for linearization

process:
    input the partial tree
    instantiate all nodes as states
    iterate over states until no states left (last one should be complete):
        if state is complete, try to advance dot of parent
            if successful, state is done
        else, try to find daughter to advance dot
            if successful, daughter is done
        
'''
from collections import defaultdict, deque
from copy import deepcopy
import sys
import psutil
from . import utils


sys.setrecursionlimit(40000)

def run(partial, verbose=1, *args, **kwargs):
    '''
    run the dynamic program. 
    iterate over states, combining children with parents
    finish when last parent is complete
    '''
    state_map = {state._id:state for state in partial.enumerate_states(version='dp')}
    stk = deque(state_map.values())
    top_listeners = defaultdict(lambda: None)
    bottom_listeners = defaultdict(lambda: None)
    for state in stk:
        if state.completed:
            bottom_listeners[state._id] = state
        else:
            top_listeners[state.next_dot] = state
    counter = 0
    utils.logprint(2, "Starting the dynamic program")
    while stk:
        #if psutil.virtual_memory().percent > 90.:
        #    raise utils.MemoryException("search explosion")
        counter += 1
        utils.logprint(3, "iteration<{}> -- stack<{}>".format(counter, len(stk)))
        if counter > 10**4:
            print('crap')
            import pdb
            pdb.set_trace()
        # get next thing off stack
        state = stk.popleft()
        
        # this skips states that were advanced by a parent being popped
        if state.advanced:
            continue
        
        # it is a child that could progress a dot
        if state.completed and state._id == partial._id:
            utils.logprint(1, "Decoding....\r")
            #sys.stdout.flush()
            results = utils.prefix_decoding(state, state_map)
            utils.logprint(1, "{} solutions".format(len(results[0])))
            assert len(results[0]) > 0
            return results
        
        if state.completed:
            utils.logprint(3, 'firing completed state<{}>'.format(len(state.position_indices)))
            # try to get a parent. if it's listening for this state, I think it's guaranteed to have it as its next dot
            parent = top_listeners[state._id] # top are those from above listening for state
            # the None check is for parents who aren't ready for this state
            if parent is not None and parent.next_dot == state._id:
                utils.logprint(3, 'with parent<{}>'.format(len(parent.inner_states)))
                # do aggressive memory maintenance
                #del top_listeners[state.id]
                # advanced the dot
                parent.step(state)
                utils.logprint(3, 'child stepped into parent')
                # check to see if this parent is now a complete subtree
                if parent.completed:
                    utils.logprint(3, 'parent is complete')
                    # if it is, have it listen for higher parents that want it
                    bottom_listeners[parent._id] = parent
                else:
                    utils.logprint(3, 'parent still needs daughter')
                    # otherwise, listen for the next child
                    top_listeners[parent.next_dot] = parent
            else: 
                utils.logprint(3, 'weird case.')
                # the state did not have a parent that wanted it; try again later
                stk.append(state)
        else:
            utils.logprint(3, 'advancing parent<{}>'.format(len(state.inner_states))) 
            # the state is not complete. check to see if its next child is ready
            child = bottom_listeners[state.next_dot] # bottom are those from below waiting to be used
            # there is a child waiting for this parent
            if child is not None:
                utils.logprint(3, 'with child<{}>'.format(len(child.inner_states)))
                # advance the state with the child. technically it should be discarded then, right?
                state.step(child)
                utils.logprint(3, 'parent has been stepped')
                # so, mark it as advanced so we can skip it at the beginning of the loop
                child.advanced = True
                # put the parent back on the stack so it can run again
                stk.append(state)
                utils.logprint(3, 'parent is on the stack')
                if state.completed:
                    utils.logprint(3, 'parent is done')
                    bottom_listeners[state._id] = state
                else:
                    utils.logprint(3, 'parent still needs daughters')
                    top_listeners[state.next_dot] = state
                #del bottom_listeners[child.id]
                #del child
            else:
                utils.logprint(3, 'child is none. weird case')
                stk.append(state)
    ## all solutions in root node and accompanied by fully specified gorn addresses
    root_node = bottom_listeners[partial._id]
    try:
        assert root_node is not None
    except:
        print("Root node is None; aka, it wasn't finished.  Investigate.")
        import pdb
        pdb.set_trace()
    
    utils.logprint(1, "Decoding....\r")
    #sys.stdout.flush()
    # try:
    results = utils.prefix_decoding(root_node, state_map)
    # except:
    #     utils.logprint(1, "no solutions (something broke)")
    #     return []
    utils.logprint(1, "{} solutions".format(len(results[0])))
    #assert len(results[0]) > 0
    return results
    
    


        
