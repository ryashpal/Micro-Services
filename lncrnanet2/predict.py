import numpy as np

import tensorflow as tf

from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import LSTM, Dropout, Dense, Input, Activation, Masking, Lambda
from keras.layers import dot, Concatenate

from keras.utils import pad_sequences

def nn(learning_rate=0.001):
    
    inD=4
    outD=2
    orfD=2
    orfref=10000
    dropout=0.5
    hidden=100

    RNN=LSTM

    rnn_input=Input(shape=(None,inD))
    orf_input=Input(shape=(None,orfD))

    #orf
    orf_size=Lambda(lambda x: K.expand_dims(K.sum(x,axis=-2)[:,0]/orfref,axis=-1), output_shape=lambda  s: (s[0],1))(orf_input)#.repeat(1)
    orf_ratio=Lambda(lambda x: K.sum(x,axis=-1),output_shape=lambda s: (s[0],s[1]))(rnn_input)
    orf_ratio=Lambda(lambda x: orfref/(K.sum(x,axis=-1,keepdims=True)+1),output_shape=lambda s: (s[0],1))(orf_ratio)

    #orf_ratio=merge([orf_size,orf_ratio],mode='dot')
    #https://stackoverflow.com/questions/52542275/merging-layers-on-keras-dot-product/52542847
    orf_ratio = dot([orf_size,orf_ratio], axes=1, normalize=False)

    orf_in=Masking()(orf_input)
    rnn_in=Masking()(rnn_input)

    orf_in=RNN(hidden,return_sequences=True)(orf_in)
    rnn_in=RNN(hidden,return_sequences=True)(rnn_in)

    #rnn_in=merge([orf_in,rnn_in],mode='concat')
    rnn_in = Concatenate(axis=-1)([orf_in,rnn_in])

    rnn_in=RNN(hidden,return_sequences=False) (rnn_in)
    rnn_in=Dropout(dropout)(rnn_in)

    # rnn_in=merge([rnn_in,orf_size,orf_ratio],mode='concat')
    rnn_in = Concatenate(axis=-1)([rnn_in,orf_size,orf_ratio])
    rnn_out=Dense(outD)(rnn_in)
    rnn_act=Activation('softmax')(rnn_out)

    model=Model(inputs=[rnn_input,orf_input],outputs=rnn_act)

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def predict_lncrnanet2(seq):
    if len(seq) < 200:
        return "Sequence length less than 200"
    elif len(seq) > 3000:
        return "Sequence length more than 3000"
    else:
        orf=[]
        seq = seq.replace('U', 'T')
        seq = char_to_int(seq)
        results = findORF(seq)
        start=results[0]
        stop=results[1]
        cur_orf=np.ones(stop-start)
        cur_orf=np.concatenate([2*np.ones(start),cur_orf,2*np.ones(len(seq)-stop)],axis=0)
        orf.append(cur_orf)
        orf = np.array(orf)
        orf = pad_sequences(orf, maxlen=(int(len(orf[0])/500) + 1)*500)
        orf = (np.arange(orf.max()+1) == orf[:,:,None]).astype(dtype='float32')
        orf = np.delete(orf,0,axis=-1)
        X = pad_sequences([seq], maxlen=(int(len(seq)/500) + 1)*500)
        X = (np.arange(X.max()+1) == X[:,:,None]).astype(dtype='float32')
        X = np.delete(X,0,axis=-1)
        model={}
        model['net']=nn(learning_rate=0.001)
        model['optimizer'] = tf.keras.optimizers.Adam(learning_rate=0.001)
        model['loss_fn'] = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        model['checkpoint'] = tf.train.Checkpoint(model['net'])
        model['checkpoint'].restore('data/checkpoint-200')
        pred = model['net'].predict([X, orf], batch_size=512)
        return pred[0][0]


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


#!/usr/bin/python3

import os
import sys
import numpy as np

#keras
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import load_model
from keras.utils import pad_sequences

sys.setrecursionlimit(100000)

def findORF(seq):
    orflen=0
    orf=""
    o_s=0
    o_e=0
    length=len(seq)
    seq=[seq]
    seq= pad_sequences(seq,maxlen=length,padding='post')
    seq=(np.arange(seq.max()+1) == seq[:,:,None]).astype(dtype='float32')
    seq=np.delete(seq,0,axis=-1)
    for frame in range(3):
        tseq=stopmodel.predict(seq[:,frame:])[:,:(length-frame)//3]
        tseq=np.argmax(tseq,axis=-1)-1
        sseq=np.append(-1,np.where(tseq==1)[1])
        sseq=np.append(sseq,tseq.shape[1])
        lseq=np.diff(sseq)-1
        flenp=np.argmax(lseq)
        flen=lseq[flenp]
        n_s=frame+3*sseq[flenp]+3
        n_e=frame+3*sseq[flenp+1]
        
        if flen>orflen or ((orflen==flen) and n_s<o_s):
            orflen=flen
            o_s=n_s
            o_e=n_e

    return o_s,o_e

stopmodel=load_model('data/stopfinder_singleframe.h5')


if __name__ == '__main__':
    predict_lncrnanet2('CAGCGCTTGGGGCTCGCGGGCCGCTCCCTCCGCTCGGAAGGGAAAAGTCTGAAGACGCTTATGTCCAAGGGGATCCTGCAGGTGCATCCTCCGATCTGCGACTGCCCGGGCTGCCGAATATCCTCCCCGGTGAACCGGGGGCGGCTGGCAGACAAGAGGACAGTCGCCCTGCCTGCCGCCCGGAACCTGAAGAAGGAGCGAACTCCCAGCTTCTCTGCCAGCGATGGTGACAGCGACGGGAGTGGCCCCACCTGTGGGCGGCGGCCAGGCTTGAAGCAGGAGGATGGTCCGCACATCCGTATCATGAAGAGAAGAGTCCACACCCACTGGGACGTGAACATCTCTTTCCGAGAGGCGTCCTGCAGCCAGGACGGCAACCTTCCCACC')
