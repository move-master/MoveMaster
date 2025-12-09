import sys
import copy

height = 0

def genStartTwr(height):
    result = "1" * 3 * height
    return result

def printTwr(s : str):
    count = 0
    for char in s:
        print(char,end="")
        count += 1
        if count % 3 == 0:
            print(" ",end="")

def getSuccessorStates(currState : str) -> list:
    successorLst = []
    blockCutoff = 3
    state = list(currState)
    for i in range(len(state)):
        state[i] = int(state[i])
    
    if state[-3:] == [1,1,1]: # If top layer complete
        print("!!!")
        state += [0,0,0]
        blockCutoff += 3

    stateCopy = copy.deepcopy(state)

    for i in range(len(state)-blockCutoff): # [1,1,1,1,1,1, | 1,1,1,0,0,0]
        for j in range(len(state)-3,len(state)):
            state = stateCopy
            print(state)
            if state[i] == 1 and state[j] == 0:
                state[i], state[j] = state[j], state[i]
                if state in successorLst:
                    continue
                else:
                    successorLst.append(state)
    return successorLst


if len(sys.argv) != 2: # sys.argv[0] == script name, sys.argv[1] == height.
    print("ERROR: Please include height.")
    exit
else:
    height = 3 # sys.argv[1]
    print(f"height: {height}")


if __name__ == "__main__":
    tower = genStartTwr(height)
    printTwr(tower)
    print(getSuccessorStates(tower))
