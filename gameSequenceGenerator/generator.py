import sys
import random
import math

print("running game sequence generator...\n")

class Block:
    def __init__(self, ID, pos):
        self.ID = int(ID)
        self.pos = pos

    def __repr__(self):
        if self.ID == -1:
            return "<   >"
        if self.ID < 10:
            return f"<B0{self.ID}>"
        return f"<B{self.ID}>"

    def __eq__(self, other):
        return (self.ID == other.ID)

    def __hash__(self):
        return hash(self.ID)

    def getID(self):
        return self.ID

    def getPos(self):
        return self.pos

    def setID(self, ID):
        self.ID = ID

    def setPos(self, pos):
        self.pos = pos

    def isNull(self):
        return (self.ID == -1)


class Tower:
    def __init__(self, name):
        self.name = str(name)
        self.moves = 0
        self.tower = []
        blockID = 1
        for _layer in range(18):
            row = []
            for _ in range(3):
                row.append(Block(blockID, 0))
                blockID += 1
            self.tower.append(row)

    def __repr__(self):
        out = ""
        for i in range(len(self.tower) - 1, -1, -1):
            out += "\n" + str(self.tower[i])
        return f":{self.name}:{out}"

    def _findBlock(self, ID):
        for l_idx, layer in enumerate(self.tower):
            for b_idx, blk in enumerate(layer):
                if blk.getID() == ID:
                    return (l_idx, b_idx)
        return (None, None)

    def isTowerValid(self):
        if len(self.tower) < 18 or len(self.tower) > 54:
            return False
        blockSet = set([-1])
        for layer in self.tower:
            for blk in layer:
                if blk.getID() != -1 and (blk.getID() < 1 or blk.getID() > 54):
                    return False
                blockSet.add(blk.getID())
        if len(blockSet) != 55:
            return False
        return True

    def move(self, ID, newPos):
        """
        Pure structural move (no probabilities, no RL hooks).
        - Remove block with ID from its layer.
        - Place on top (existing top if it has an empty slot; else create a new layer).
        - Return True if the structural move succeeds, False if it is illegal (e.g., target top slot occupied or block not found).
        """
        lvl, idx = self._findBlock(ID)
        if lvl is None:
            return False

        self.tower[lvl][idx].setID(-1)

        topLayer = self.tower[-1]
        hasNull = any(b.getID() == -1 for b in topLayer)
        if hasNull:
            if not topLayer[newPos - 1].isNull():
                return False
            topLayer[newPos - 1] = Block(ID, newPos)
        else:
            newLayer = [Block(-1, 1), Block(-1, 2), Block(-1, 3)]
            newLayer[newPos - 1] = Block(ID, newPos)
            self.tower.append(newLayer)

        self.moves += 1
        return True

    def generateSequence(self, max_moves=150):
        """
        Generate a structural sequence only (no probability failures).
        Stops when:
          - the tower becomes invalid structurally, or
          - max_moves is exceeded.
        """
        sequence = ""
        pos_cycle = [1, 2, 3]
        while True:
            if len(pos_cycle) == 0:
                pos_cycle = [1, 2, 3]
            randPos = random.choice(pos_cycle)
            pos_cycle.remove(randPos)

            topIDs = set(b.getID() for b in self.tower[-1])
            randID = 0
            while randID == 0 or randID in topIDs:
                randID = random.randint(1, 54)

            ok = self.move(randID, randPos)
            if randID < 10:
                sequence += "0"
            sequence += f"{randID}.{randPos} "

            if not ok:
                return sequence

            if not self.isTowerValid():
                return sequence

            if self.moves > max_moves:
                return sequence


def isValidSequence(_seq):
    return True


seqSet = set()
seqLenLst = []
SEQ_GEN_NUM = 10000
runGenerator = False

if runGenerator:
    print(f"Generating {SEQ_GEN_NUM} sequences...")
    i = 0
    with open(f"{SEQ_GEN_NUM}-sequences.txt", "w") as file:
        while i < SEQ_GEN_NUM:
            tower = Tower("start")
            seq = tower.generateSequence(max_moves=150)
            seqSet.add(seq)
            seqLenLst.append(len(seq.split()))
            i += 1
        for s in seqSet:
            file.write(s + "\n")
    print("...done.\n")
    print("Min seq length:", min(seqLenLst))
    print("Max seq length:", max(seqLenLst))
