#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: spectral.py
#
# ------------------------------------
"""spectral: Compute spectral coefficients

Based on cmusphinx.mfcc from the Sphinx-III source code
Original author: David Huggins-Daines
Original copyright (c) 2006 Carnegie Mellon University
Modifications by Maarten Versteegh and Michele Gubian

License: BSD style

"""

from __future__ import division

import numpy as np
import scipy.signal

from _logspec import pre_emphasis
import scales

class Spectral(object):
    """
    Extract spectral features from an audio signal.

    Parameters
    ----------
    scale : {'mel', 'bark', 'erb'}
        Perceptual scale.
    nfilt : int, optional
        Number of filters.
    do_dct : bool, optional
        Perform Discrete Cosine Transform.
    ncep : int, optional
        Number of cepstra, only applicable if `do_dct` is True.
    lowerf : float, optional
        Lowest frequency of the filterbank.
    upperf : float, optional
        Highest frequency of the filterbank.
    alpha : float, optional
        Pre-emphasis coefficient.
    fs : int, optional
        Sampling rate.
    wlen : float, optional
        Window length.
    nfft : int, optional
        Length of the DFT.
    compression : {'log', 'cubic'}
        Type of amplitude compression.
    do_deltas : bool, optional
        Calculate 1st derivative (speed)
    do_deltasdeltas : bool, optional
        Calculate 2nd derivative (acceleration)


    Methods
    -------
    transform(sig)
        Encode a signal.

    """
    def __init__(self,
                 scale='mel',           # perceptual scaling of the filterbanks
                 nfilt=40,              # number of filters in mel bank
                 ncep=13,               # number of cepstra
                 do_dct=True,           # perform DCT
                 lowerf=133.3333,       # bottom frequency of the filterbanks
                 upperf=6855.4976,      # top frequency of the filterbanks
                 alpha=0.97,            # pre-emphasis coefficient
                 fs=16000,              # sampling rate
                 frate=100,             # framerate
                 wlen=0.01,             # window length
                 nfft=512,              # length of dft
                 compression='log',     # amplitude compression
                 do_deltas=False,       # calculate 1st derivative (speed)
                 do_deltasdeltas=False  # calculate 2nd derivative (acceleration)
                 ):
        if not nfilt > 0:
            raise(ValueError,
                  'Number of filters must be positive, not {0:%d}'
                  .format(nfilt))
        if upperf > fs // 2:
            raise(ValueError,
                  "Upper frequency %f exceeds Nyquist %f" % (upperf, fs // 2))
        compression_types = ['log', 'cubicroot']
        if not compression in compression_types:
            raise(ValueError,
                  'Compression must be one of [{0:s}], not {1}'
                  .format(', '.join(compression_types), compression))
        self.compression = compression

        self._set_scale(scale)
        self.lowerf = lowerf
        self.upperf = upperf
        self.nfft = nfft
        self.ncep = ncep
        self.do_dct = do_dct
        self.nfilt = nfilt
        self.frate = frate
        self.fs = fs
        self.fshift = fs / frate
        self.do_deltas = do_deltas
        self.do_deltasdeltas = do_deltasdeltas
        self.wlen_secs = wlen

        # build hamming window
        self.wlen_samples = int(wlen * fs)
        self.win = np.hamming(self.wlen_samples)

        # prior sample for pre-emphasis
        self.prior = 0
        self.alpha = alpha

        self._build_filters()

    def _set_scale(self, scale):
        self.scale = scale
        if scale == 'mel':
            self.to_hertz = scales.mel_to_hertz
            self.from_hertz = scales.hertz_to_mel
        elif scale == 'bark':
            self.to_hertz = scales.bark_to_hertz
            self.from_hertz = scales.hertz_to_bark
        elif scale == 'erb':
            self.to_hertz = scales.erb_to_hertz
            self.from_hertz = scales.hertz_to_erb
        else:
            raise ValueError('scale must be one of [{0}], not {1}'.format(
                ', '.join(['mel', 'bark', 'erb']), scale))

    def _build_filters(self):
        # build mel filter matrix
        self.filters = np.zeros((self.nfft//2 + 1, self.nfilt), 'd')
        dfreq = self.fs / self.nfft

        melmax = self.from_hertz(self.upperf)
        melmin = self.from_hertz(self.lowerf)
        dmelbw = (melmax - melmin) / (self.nfilt + 1)
        # filter edges in hz
        filt_edge = self.to_hertz(melmin + dmelbw *
                                  np.arange(self.nfilt + 2, dtype='d'))

        for whichfilt in range(0, self.nfilt):
            # Filter triangles in dft points
            leftfr = round(filt_edge[whichfilt] / dfreq)
            centerfr = round(filt_edge[whichfilt + 1] / dfreq)
            rightfr = round(filt_edge[whichfilt + 2] / dfreq)

            fwidth = (rightfr - leftfr) * dfreq
            height = 2 / fwidth

            if centerfr != leftfr:
                leftslope = height / (centerfr - leftfr)
            else:
                leftslope = 0
            freq = leftfr + 1
            while freq < centerfr:
                self.filters[int(freq), whichfilt] = \
                    (freq - leftfr) * leftslope
                freq += 1
            if freq == centerfr:
                self.filters[int(freq), whichfilt] = height
                freq += 1
            if centerfr != rightfr:
                rightslope = height / (centerfr - rightfr)
            while freq < rightfr:
                self.filters[int(freq), whichfilt] = \
                    (freq - rightfr) * rightslope
                freq += 1
        if self.do_dct:
            self.s2dct = s2dctmat(self.nfilt, self.ncep, 1/self.nfilt)

    @property
    def config(self):
        return dict(nfilt=self.nfilt,
                    ncep=self.ncep,
                    do_dct=self.do_dct,
                    lowerf=self.lowerf,
                    upperf=self.upperf,
                    alpha=self.alpha,
                    fs=self.fs,
                    frate=self.frate,
                    wlen=self.wlen_secs,
                    nfft=self.nfft,
                    compression=self.compression,
                    do_deltas=self.do_deltas,
                    do_deltasdeltas=self.do_deltasdeltas)

    def compressor(self, spec):
        if self.compression == 'log':
            return np.log(spec)
        elif self.compression == 'cubicroot':
            return spec**(1./3)


    def transform(self, sig):
        sig = sig.astype(np.double)
        nfr = int(len(sig) / self.fshift + 1)
        if self.do_dct:
            c = np.zeros((nfr, self.ncep))
        else:
            c = np.zeros((nfr, self.nfilt))
        for fr in xrange(nfr):
            start = int(round(fr * self.fshift))
            end = min(len(sig), start + self.wlen_samples)
            frame = sig[start:end]
            if len(frame) < self.wlen_samples:
                frame = np.resize(frame, self.wlen_samples)
                frame[self.wlen_samples:] = 0
            spec = self.frame2spec(frame)
            cspec = self.compressor(spec)
            if self.do_dct:
                c[fr] = np.dot(cspec, self.s2dct.T) / self.nfilt
            else:
                c[fr] = cspec
        r = c
        if self.do_deltas:
            r = np.c_[r, self.calc_deltas(c)]
        if self.do_deltasdeltas:
            r = np.c_[r, self.calc_deltasdeltas(c)]
        self.prior = 0
        return r

    def frame2spec(self, frame):
        tmp = frame[-1]
        frame = pre_emphasis(frame, self.prior, self.alpha) * self.win
        self.prior = tmp
        fft = np.fft.rfft(frame, self.nfft)
        power = fft.real * fft.real + fft.imag * fft.imag
        return np.dot(power, self.filters).clip(1e-5, np.inf)

    def calc_deltas(self, X):
        """ compute delta coefficients
        """
        nframes, nceps = X.shape
        hlen = 4
        a = np.r_[hlen:-hlen-1:-1] / 60
        g = np.r_[np.array([X[1, :] for x in range(hlen)]),
                  X,
                  np.array([X[nframes-1, :] for x in range(hlen)])]
        flt = scipy.signal.lfilter(a, 1, g.flat)
        d = flt.reshape((nframes + 8, nceps))
        return np.array(d[8:, :])

    def calc_deltasdeltas(self, X):
        nframes, nceps = X.shape
        hlen = 4
        a = np.r_[hlen:-hlen-1:-1] / 60

        hlen2 = 1
        f = np.r_[hlen2:-hlen2-1:-1] / 2

        g = np.r_[np.array([X[1, :] for x in range(hlen+hlen2)]),
                  X,
                  np.array([X[nframes-1, :] for x in range(hlen+hlen2)])]

        flt1 = scipy.signal.lfilter(a, 1, g.flat)
        h = flt1.reshape((nframes + 10, nceps))[8:, :]

        flt2 = scipy.signal.lfilter(f, 1, h.flat)
        dd = flt2.reshape((nframes + 2, nceps))
        return dd[2:, :]


def s2dctmat(nfilt, ncep, freqstep):
    melcos = np.empty((ncep, nfilt), 'double')
    for i in range(0, ncep):
        freq = np.pi * i / nfilt
        melcos[i] = np.cos(freq * np.arange(0.5, nfilt + 0.5, 1.0, 'double'))
    melcos[:, 0] *= 0.5
    return melcos


def dctmat(N, K, freqstep, orthogonalize=True):
    """Return the orthogonal DCT-II/DCT-III matrix of size NxK.
    For computing or inverting MFCCs, N is the number of
    log-power-spectrum bins while K is the number of ceptra."""
    cosmat = np.zeros((N, K), 'double')
    for n in range(N):
        for k in range(K):
            cosmat[n, k] = np.cos(freqstep * (n + 0.5) * k)
    if orthogonalize:
        cosmat[:, 0] = cosmat[:, 0] * 1/np.sqrt(2)
    return cosmat
