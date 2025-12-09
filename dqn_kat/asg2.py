import copy

class InitializationError(Exception):
    pass

# Tower objects represent jenga tower states. A basic check is performed to ensure that the initial state length is divisible by 3,
class Tower:
    def __init__(self, state : tuple):
        self.state = state
        if len(state) % 3 != 0:
            raise InitializationError("Length of state should be divisible by 3.")
        
    def __repr__(self):

        result = f"Tower({self.state})"
        return result
    
    def __eq__(self, other):
        if self.state == other.state:
            return True
        else:
            return False
        
    def __hash__(self):
        return hash(self.state)
    
    # Prints a "pretty" visualization of the tower, where each layer is represented as a list of three elements.
    # Printed from the top down, so last layer corresponds to bottom of the tower.
    # Layers printed from left to right, as they are initialized in tower.state.
    def pretty(self):
        counter = 0
        layer = []
        for block in reversed(self.state):
            layer.insert(0, block)
            counter += 1
            if counter == 3:
                counter = 0
                print(layer)
                layer = []

# Returns a list of three bits that correspond to a layer on the tower at index == num.
def getLayer(tower : Tower, num : int):
    counter = 0
    groupOf3 = []
    layers = []
    for block in tower.state:
        groupOf3.append(block)
        counter += 1
        if counter == 3:
            counter = 0
            layers.append(groupOf3)
            groupOf3 = []
    return layers[num]

# Basic validator for Jenga tower states. It checks to ensure that the tower is theoretically stable, has no empty layers, and the
# layer second from the top is complete (under the official rules of Jenga, this layer should always be complete at the start of each turn).
# Additionally, checks to make sure that the total number of blocks in the tower is divisible by 3.
# The validator is not capable of checking for invalid/impossible state transitions, however.
def isValid(tower : Tower, flag : bool):
    height = len(tower.state)//3
    result = False
    totalBlocks = 0
    for layer in range(height):
        currLayer = getLayer(tower, layer)
        for block in currLayer:
            totalBlocks += block
        if currLayer == [0,0,1] or currLayer == [1,0,0]:
            if layer == height-1:
                result = True
            else:
                if(flag): {print(f"INVALID b/c encountered unstable layer")}
                return False
        elif currLayer == [0,1,0] or currLayer == [1,0,1] or currLayer == [0,1,1] or currLayer == [1,1,0]:
            if layer != height-2:
                result = True
            else:
                if(flag): {print(f"INVALID b/c cannot remove blocks from layer directly below incomplete top layer")}
                return False
        elif currLayer == [0,0,0]:
            if(flag): {print(f"INVALID b/c cannot have entirely empty layer.")}
            return False
        else:
            result = True
    if totalBlocks % 3 != 0:
        if(flag): {print(f"INVALID b/c totalBlocks is {totalBlocks}, which is NOT divisible by 3.")}
        return False
    return result

def getSuccessors(tower : Tower, succSet : set, rd : int):
    successors = []
    state = list(tower.state)
    if state[-3:] == [1,1,1]:
        state.extend([0,0,0])
    originalState = copy.deepcopy(state)
    for i in range(len(state)-6):
        state = copy.deepcopy(originalState)
        for j in range(len(state)-3,len(state),1):

            state = copy.deepcopy(originalState)
            if state[j] == 0:
                state[i], state[j] = state[j], state[i]
                sTower = Tower(tuple(state))
                if sTower in succSet:
                    continue
                if isValid(sTower, False):
                    successors.append(sTower)
                    succSet.add(sTower)
    if len(successors) == 0:
        return
    for s in successors:
        getSuccessors(s, succSet, rd+1)


    



if __name__ == "__main__":
    print("Running asg2.py...")
    
    # The following state, while it passes the validator, is not possible to achieve through legal Jenga moves.
    state = (0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0)
    """
    tower = Tower(state)
    tower.pretty()
    print(f" is tower valid? >> {isValid(tower, True)}")
    totalBlocks = 0
    for block in state:
        totalBlocks += block
    print(F"num blocks: {totalBlocks}")
    print(F"height: {len(state)//3}")
    """

    # Start state (for classic Jenga)
    #startState = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
    startState = (1,1,1,1,1,1)
    startTower = Tower(startState)
    #startTower.pretty()
    #print(f" is tower valid? >> {isValid(startTower, True)}")
    totalBlocks = 0
    for block in startState:
        totalBlocks += block
    #print(F"num blocks: {totalBlocks}")
    #print(F"height: {len(startState)//3}")
    
    mySet = set()
    getSuccessors(startTower, mySet, 0)
    for state in mySet:
        print(state)