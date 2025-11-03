
def layer_com(layer: list):
    if layer == [0,1,1]:
        return 1
    elif layer == [1,1,0]:
        return -1
    elif layer == [1,0,0]:
        return -1
    elif layer == [0,0,1]:
        return 1
    else:
        return 0

def com(state: list):
    #rep com as list [x,y,z]
    #weight = 1 per block
    com = [0,0,0]
    #start with z:
    add = 0
    total = 0
    for i in range(len(state)//3):
        add += 1
        for j in range(3):
            if state[3*i+j] == 1:
                total += add
    com[2] = total / 54

    x = [0,1,2,6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38,42,43,44,48,49,50,54,55,56,60,61,62,66,67,68,72,73,74]
    #y = [3,4,5,9,10,11,15,16,17,21,22,23,27,28,29,33,34,35,39,40,41,45,46,47,51,52,53]

    for i in range(len(state)//3):
        layer = layer_com([state[3*i],state[3*i+1],state[3*i+2]])
        if 3*i in x:
            com[0] += layer
        else:
            com[1] += layer 

    com[0] = com[0] / (len(state)//3)
    com[1] = com[1] / (len(state)//3)
    return com


