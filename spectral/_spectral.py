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

import abc

import numpy as np
import scipy.signal

from _logspec import pre_emphasis


def hertz_to_mel(f):
    """Convert frequency in Hertz to mel.

    :param f: frequency in Hertz
    :return: frequency in mel
    """
    return 2595. * np.log10(1.+f/700)


def mel_to_hertz(m):
    """Convert frequency in mel to Hertz.

    :param m: frequency in mel
    :return: frequency in Hertz
    """
    return 700. * (np.power(10., m/2595) - 1.)


class Spectral(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def transform(self, signal):
        """Return the spectral transform of \c signal.

        :return:
          Iterable of ndarrays(n_frames, n_features). Typically, the iter
          will consist of the static, delta and double-delta spectral features.
        """
        return

    @abc.abstractproperty
    def config(self):
        """Return a dict of configuration values."""
        return


class Mel(Spectral):
    def __init__(self, **kwargs):
        self._s = _Spec(**kwargs)

    def transform(self, sig):
        s = self._s.sig2logspec(sig)
        nfilt = self._s.config['nfilt']
        ds = self._s.config['mel_deltas']
        dds = self._s.config['mel_deltasdeltas']
        r = [s[:, :nfilt]]
        if ds:
            r.append(s[:, nfilt:2*nfilt])
        if dds:
            r.append(s[:, 2*nfilt:3*nfilt])
        return r

    @property
    def config(self):
        return self._s.config


class MFCC(Spectral):
    def __init__(self, **kwargs):
        self._s = _Spec(**kwargs)

    def transform(self, sig):
        s = self._s.mfcc(sig)
        nceps = self._s.config['ncep']
        ds = self._s.config['mfcc_deltas']
        dds = self._s.config['mfcc_deltasdeltas']
        r = [s[:, :nceps]]
        if ds:
            r.append(s[:, nceps:2*nceps])
        if dds:
            r.append(s[:, 2*nceps:3*nceps])
        return r

    @property
    def config(self):
        return self._s.config


class _Spec(object):
    def __init__(self,
                 nfilt=40,              # number of filters in mel bank
                 ncep=13,               # number of cepstra
                 lowerf=133.3333,
                 upperf=6855.4976,
                 alpha=0.97,            # pre-emphasis coefficient
                 fs=16000,              # sampling rate
                 frate=100,
                 wlen=0.01,             # window length
                 nfft=512,              # length of dft
                 mfcc_deltas=True,
                 mfcc_deltasdeltas=True,
                 mel_deltas=True,
                 mel_deltasdeltas=True
                 ):
        # store params
        self.lowerf = lowerf
        self.upperf = upperf
        self.nfft = nfft
        self.ncep = ncep
        self.nfilt = nfilt
        self.frate = frate
        self.fs = fs
        self.fshift = fs / frate
        self.mfcc_deltas = mfcc_deltas
        self.mfcc_deltasdeltas = mfcc_deltasdeltas
        self.mel_deltas = mel_deltas
        self.mel_deltasdeltas = mel_deltasdeltas

        self.wlen_secs = wlen
        # build hamming window
        self.wlen_samples = int(wlen * fs)
        self.win = np.hamming(self.wlen_samples)

        # prior sample for pre-emphasis
        self.prior = 0
        self.alpha = alpha

        # build mel filter matrix
        self.filters = np.zeros((nfft//2 + 1, nfilt), 'd')
        dfreq = fs / nfft
        if upperf > fs // 2:
            raise(Exception,
                  "Upper frequency %f exceeds Nyquist %f" % (upperf. fs // 2))
        melmax = hertz_to_mel(upperf)
        melmin = hertz_to_mel(lowerf)
        dmelbw = (melmax - melmin) / (nfilt + 1)
        # filter edges in hz
        filt_edge = mel_to_hertz(melmin + dmelbw *
                                 np.arange(nfilt + 2, dtype='d'))

        for whichfilt in range(0, nfilt):
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

        self.s2dct = s2dctmat(nfilt, ncep, 1/nfilt)
        self.dct = dctmat(nfilt, ncep, np.pi/nfilt)

    def __eq__(self, other):
        if type(other) != MFCC:
            return False
        if self.config() != other.config():
            return False
        return True

    @property
    def config(self):
        return dict(nfilt=self.nfilt,
                    ncep=self.ncep,
                    lowerf=self.lowerf,
                    upperf=self.upperf,
                    alpha=self.alpha,
                    fs=self.fs,
                    frate=self.frate,
                    wlen=self.wlen_secs,
                    nfft=self.nfft,
                    mfcc_deltas=self.mfcc_deltas,
                    mfcc_deltasdeltas=self.mfcc_deltasdeltas,
                    mel_deltas=self.mel_deltas,
                    mel_deltasdeltas=self.mel_deltasdeltas)

    def mfcc(self, sig, deltas=None, deltasdeltas=None):
        if deltas is None:
            deltas = self.mfcc_deltas
        if deltasdeltas is None:
            deltasdeltas = self.mfcc_deltasdeltas
        sig = sig.astype(np.double)
        nfr = int(len(sig) / self.fshift + 1)
        mfcc = np.zeros((nfr, self.ncep), 'd')
        fr = 0
        # loop over frames
        while fr < nfr:
            start = int(round(fr * self.fshift))
            end = int(min(len(sig), start + self.wlen_samples))
            frame = sig[start:end]
            if len(frame) < self.wlen_samples:
                frame = np.resize(frame, self.wlen_samples)
                frame[self.wlen_samples:] = 0
            mfcc[fr] = self.frame2s2mfc(frame)
            fr += 1
        c = mfcc
        if deltas:
            d = self.deltas(mfcc)
            c = np.c_[c, d]
        if deltasdeltas:
            dd = self.deltasdeltas(mfcc)
            c = np.c_[c, dd]
        return c

    def sig2logspec(self, sig, deltas=None, deltasdeltas=None):
        if deltas is None:
            deltas = self.mfcc_deltas
        if deltasdeltas is None:
            deltasdeltas = self.mfcc_deltasdeltas
        sig = sig.astype(np.double)
        nfr = int(len(sig) / self.fshift + 1)
        mfcc = np.zeros((nfr, self.nfilt), 'd')
        fr = 0
        while fr < nfr:
            start = round(fr * self.fshift)
            end = min(len(sig), start + self.wlen_samples)
            frame = sig[start:end]
            if len(frame) < self.wlen_samples:
                frame = np.resize(frame, self.wlen_samples)
                frame[self.wlen_samples:] = 0
            mfcc[fr] = self.frame2logspec(frame)
            fr += 1
        c = mfcc
        if deltas:
            d = self.deltas(mfcc)
            c = np.c_[c, d]
        if deltasdeltas:
            dd = self.deltasdeltas(mfcc)
            c = np.c_[c, dd]
        return c

    # def pre_emphasis(self, frame):
    #     outfr = np.empty(len(frame), 'd')
    #     outfr[0] = frame[0] - self.alpha * self.prior
    #     for i in range(1, len(frame)):
    #         outfr[i] = frame[i] - self.alpha * frame[i-1]
    #     self.prior = frame[-1]
    #     return outfr

    def frame2logspec(self, frame):
        tmp = frame[-1]
        frame = pre_emphasis(frame, self.prior, self.alpha) * self.win
        self.prior = tmp
        fft = np.fft.rfft(frame, self.nfft)
        power = fft.real * fft.real + fft.imag * fft.imag
        return np.log(np.dot(power, self.filters).clip(1e-5, np.inf))

    def frame2s2mfc(self, frame):
        logspec = self.frame2logspec(frame)
        return np.dot(logspec, self.s2dct.T) / self.nfilt

    def deltas(self, cepstra):
        """ compute delta coefficients of mfccs
        """
        nframes, nceps = cepstra.shape
        hlen = 4
        a = np.r_[hlen:-hlen-1:-1] / 60
        g = np.r_[np.array([cepstra[1, :] for x in range(hlen)]),
                  cepstra,
                  np.array([cepstra[nframes-1, :] for x in range(hlen)])]
        flt = scipy.signal.lfilter(a, 1, g.flat)
        d = flt.reshape((nframes + 8, nceps))
        return np.array(d[8:, :])

    def deltasdeltas(self, cepstra):
        nframes, nceps = cepstra.shape
        hlen = 4
        a = np.r_[hlen:-hlen-1:-1] / 60

        hlen2 = 1
        f = np.r_[hlen2:-hlen2-1:-1] / 2

        g = np.r_[np.array([cepstra[1, :] for x in range(hlen+hlen2)]),
                  cepstra,
                  np.array([cepstra[nframes-1, :] for x in range(hlen+hlen2)])]

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
