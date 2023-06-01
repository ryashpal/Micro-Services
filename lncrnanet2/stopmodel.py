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
