#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: wav2mfcc.py
# date: Fri February 07 22:24 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""wav2mfcc: convert wave file args[1] to mfcc txt file args[2]

"""

from __future__ import division

from spectral import MFCC
import wave
import struct
import sys
import numpy as np

input = sys.argv[1]
output = sys.argv[2]

# load wave file as numpy array, needs adjustment for multichannel files
fid = wave.open(input, 'r')
_, _, fs, nframes, _, _ = fid.getparams()
sig = np.array(struct.unpack_from("%dh" % nframes, fid.readframes(nframes)))
fid.close()

# convert to mfccs
mfcc = MFCC(nfilt=40,               # number of filters in mel bank
            ncep=13,                # number of cepstra
            alpha=0.97,             # pre-emphasis
            fs=fs,                  # sampling rate
            frate=100,              # frame rate
            wlen=0.01,              # window length
            nfft=512,               # length of dft
            mfcc_deltas=True,       # also return speed
            mfcc_deltasdeltas=True  # also return acceleration
            )
feats = mfcc.transform(sig)
# if deltas are used, stack them all up
if len(feats) > 1:
    feats = np.hstack(feats)
np.savetxt(output, feats)
