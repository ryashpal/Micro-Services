import numpy as np


def char_to_int(seq):
    chars="0AGCT"
    ctable = CharacterTable(chars)
    return ctable.encode(seq)


class CharacterTable(object): #make encoding table
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    #chars : 0 (padding ) + other characters
    '''
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, l):
        X = np.zeros((len(l)),dtype=int)
        for i, c in enumerate(l):
            X[i]= self.char_indices[c]
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)
