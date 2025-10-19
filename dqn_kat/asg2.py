class InitializationError(Exception):
    pass

class Tower:
    def __init__(self, height : int, state : tuple):
        self.height = height
        self.state = state
        if len(state) % 3 != 0:
            raise InitializationError("Length of state should be divisible by 3.")
        elif height*3 != len(state):
            raise InitializationError("Length of state must be equal to height*3.")

    def __repr__(self):

        result = f"Tower({self.height}, {self.state})"
        return result
# Initialize a tuple with the value 'hello' repeated 5 times


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
    
def isValid(tower : Tower, flag : bool):
    result = False
    totalBlocks = 0
    topLayer = getLayer(tower, tower.height-1)
    for layer in range(tower.height):
        currLayer = getLayer(tower, layer)
        for block in currLayer:
            totalBlocks += block
        print(f"currLayer == {currLayer}")
        if currLayer == [0,0,1] or currLayer == [1,0,0] or currLayer == [0,0,0]:
            if layer == tower.height-1:
                result = True
            else:
                if(flag): {print(f"False b/c encountered unstable layer")}
                return False
        elif currLayer == [0,1,0] or currLayer == [1,0,1] or currLayer == [0,1,1] or currLayer == [1,1,0]:
            if layer != tower.height-2:
                result = True
            else:
                if(flag): {print(f"False b/c cannot remove blocks from layer directly below incomplete top layer")}
                return False
        else:
            result = True
    if totalBlocks != 54:
        if(flag): {print(f"False b/c totalBlocks is {totalBlocks}")}
        return False
    return result




if __name__ == "__main__":
    print("Running asg2.py...")
    state = (1,1,0,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1)
    height = len(state)//3
    tower = Tower(height,state)
    print(f" is tower valid? >> {isValid(tower, True)}")
