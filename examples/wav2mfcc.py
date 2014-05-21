#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: features.py
# date: Tue April 22 18:31 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""wav2mfcc: convert wave files to mfcc

"""

from __future__ import division

import argparse
import json
import os.path as path
import os
import wave
import struct

import numpy as np

import spectral


def resample(sig, ratio):
    try:
        import scikits.samplerate
        return scikits.samplerate.resample(sig, ratio, 'sinc_best')
    except ImportError:
        import scipy.signal
        return scipy.signal.resample(sig, int(round(sig.shape[0] * ratio)))


def parse_args():
    parser = argparse.ArgumentParser(
        prog='wav2mfcc.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Extract spectral features from audio files.',
        epilog="""Example usage:

$ ./wav2mfcc.py *.wav -o . -c config.json

extracts features from audio files in current directory.

The output format is binary .npy containing an array of nframes x nfeatures.
To load these files in python:

>>> import numpy as np
>>> features = np.load('/path/to/file.npy')
""")
    parser.add_argument('files', metavar='WAV',
                        nargs='+',
                        help='input audio files')
    parser.add_argument('-o', '--output',
                        action='store',
                        dest='outdir',
                        required=True,
                        help='output directory')
    parser.add_argument('-c', '--config',
                        action='store',
                        dest='config',
                        required=True,
                        help='configuration file')
    parser.add_argument('-f', '--force',
                        action='store_true',
                        dest='force',
                        default=False,
                        help='force resampling in case of samplerate mismatch')
    return vars(parser.parse_args())


def convert(files, outdir, encoder, force):
    for f in files:
        try:
            fid = wave.open(f, 'r')
            _, _, fs, nframes, _, _ = fid.getparams()
            sig = np.array(struct.unpack_from("%dh" % nframes,
                                              fid.readframes(nframes)))
            fid.close()
        except IOError:
            print 'No such file:', f
            exit()

        if fs != encoder.config['fs']:
            if force:
                sig = resample(sig, fs / encoder.config['fs'])
                # resample
                pass
            else:
                print ('Samplerate mismatch, expected {0}, got {1}, in {2}.\n'
                       'Use option -f to force resampling of the audio file.'
                       .format(encoder.config['fs'], fs, f))
                exit()

        # feats = np.hstack(encoder.transform(sig))
        feats = encoder.transform(sig)
        bname = path.splitext(path.basename(f))[0]
        np.save(path.join(outdir, bname + '.npy'), feats)


if __name__ == '__main__':
    args = parse_args()
    config_file = args['config']
    try:
        with open(config_file, 'r') as fid:
            config = json.load(fid)
    except IOError:
        print 'No such file:', config_file
        exit()

    outdir = args['outdir']
    if not os.path.exists(outdir):
        print 'No such directory:', outdir
        exit()


    encoder = _spectral._Spec(**config)

    force = args['force']
    files = args['files']
    convert(files, outdir, encoder, force)
