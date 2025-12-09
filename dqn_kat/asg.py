"""
Action Space Generator for Jenga

This program generates the entire state space for the game of Jenga.
The action space is a subset of the state space, where invalid state transitions
are "masked off" given the current state.

The generator works by calling getSuccessorStates() recursively, starting with the
default tower state [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1].
"""
import copy

lst = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

test = [1,1,1,1,1,1,1,1,1]

def getSuccessors(state, successorsSet, depth=0):
    if depth == 50:
        return successorsSet
    state = list(state)
    #print(f"len(state) >> {len(state)}")
    if(len(state) == 54):
        return successorsSet
    
    numValidBlocks = len(state)-3
    topLayer = state[-3:]
    
    # If top layer is complete
    if topLayer == [1,1,1]:
        state.extend([0,0,0])
        originalState = copy.deepcopy(state)
        for block in range(numValidBlocks):
            state = copy.deepcopy(originalState)
            for spot in range(3):
                state = copy.deepcopy(originalState)
                state[block] = 0
                state[-(3-spot)] = 1
                successorsSet.add(tuple(copy.deepcopy(state)))
    # If top layer is NOT complete
    else: # [0,1,1,1,1,1,1,1,1,1,0,0]
        numValidBlocks -= 3
        originalState = copy.deepcopy(state)
        for block in range(numValidBlocks):
            state = copy.deepcopy(originalState)
            for spot in range(3):
                if state[-(3-spot)] == 0:
                    state = copy.deepcopy(originalState)
                    state[block] = 0
                    state[-(3-spot)] = 1
                    successorsSet.add(tuple(copy.deepcopy(state)))

    successors = copy.deepcopy(successorsSet)
    for successorState in successors:
        print(successorState)
        getSuccessors(successorState, successorsSet, depth+1)

stateSpace = set()
getSuccessors(test, stateSpace)