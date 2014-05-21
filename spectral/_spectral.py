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
import warnings

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
        kwargs['do_dct'] = False
        self._s = _Spec(**kwargs)

    def transform(self, sig):
        s = self._s.sig2logspec(sig)
        nfilt = self._s.config['nfilt']
        ds = self._s.config['do_deltas']
        dds = self._s.config['do_deltasdeltas']
        r = [s[:, :nfilt]]
        if ds:
            r.append(s[:, nfilt:2*nfilt])
        if dds:
            r.append(s[:, 2*nfilt:3*nfilt])
        return r

    @property
    def config(self):
        return self._s.config


class CubicMel(Spectral):
    def __init__(self, **kwargs):
        warnings.warn('CubicMel class will be deprecated in a future release.'
                      ' Use Mel class with keyword `compression=\'cubicroot\'`'
                      ' instead.', PendingDeprecationWarning)
        kwargs['compression'] = 'cubicroot'
        kwargs['do_dct'] = False
        self._s = _Spec(**kwargs)

    def transform(self, sig):
        s = self._s.transform(sig)
        # s = np.power(self._s.sig2spec(sig), 1./3)
        nfilt = self._s.config['nfilt']
        ds = self._s.config['do_deltas']
        dds = self._s.config['do_deltasdeltas']
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
        kwargs['do_dct'] = True
        self._s = _Spec(**kwargs)

    def transform(self, sig):
        s = self._s.mfcc(sig)
        nceps = self._s.config['ncep']
        ds = self._s.config['do_deltas']
        dds = self._s.config['do_deltasdeltas']
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
                 do_dct=True,
                 lowerf=133.3333,
                 upperf=6855.4976,
                 alpha=0.97,            # pre-emphasis coefficient
                 fs=16000,              # sampling rate
                 frate=100,
                 wlen=0.01,             # window length
                 nfft=512,              # length of dft
                 compression='log',
                 do_deltas=False,
                 do_deltasdeltas=False
                 ):
        if not nfilt > 0:
            raise(Exception,
                  'Number of filters must be positive, not {0:%d}'
                  .format(nfilt))
        if upperf > fs // 2:
            raise(Exception,
                  "Upper frequency %f exceeds Nyquist %f" % (upperf. fs // 2))
        compression_types = ['log', 'cubicroot']
        if not compression in compression_types:
            raise(Exception,
                  'Compression must be one of [{0:%s}], not {1}'
                  .format(', '.join(compression_types), compression))

        # store params
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
        self.compression = compression

        self.wlen_secs = wlen
        # build hamming window
        self.wlen_samples = int(wlen * fs)
        self.win = np.hamming(self.wlen_samples)

        # prior sample for pre-emphasis
        self.prior = 0
        self.alpha = alpha

        self._build_filters()

    def _build_filters(self):
        # build mel filter matrix
        self.filters = np.zeros((self.nfft//2 + 1, self.nfilt), 'd')
        dfreq = self.fs / self.nfft

        melmax = hertz_to_mel(self.upperf)
        melmin = hertz_to_mel(self.lowerf)
        dmelbw = (melmax - melmin) / (self.nfilt + 1)
        # filter edges in hz
        filt_edge = mel_to_hertz(melmin + dmelbw *
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
