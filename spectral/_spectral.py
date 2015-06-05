#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: _spectral.py
# date: Tue April 28 11:44 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""_spectral: speech feature extraction.

this version 0.2.0 is a complete rewrite of the code. it is not meant to be
binary compatible with the previous version. since there is no original code
left, this version is relicensed to GPL.

"""

from __future__ import division

import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.signal as ss

class Spectral(object):
    """
    Parameters
    ----------
    fs : int
        samplerate. make sure all audio is in this samplerate.
    window_length : float
        length of fft windows in seconds.
    window_shift : float
        length of fft window shift in seconds.
    nfft : int
        length of dft.
    scale : string, ['mel', 'bark', 'erb']
        perceptual frequency scale
    lower_frequency : float
        lower frequency (in Hz) for the filter banks
    upper_frequency : float
        lower frequency (in Hz) for the filter banks
    nfilt : int
        number of filters in bank
    taper_filt : bool
        rescale the filter magnitude
    compression : string, ['log', 'cubicroot']
        amplitude compression type on the filter banks.
    do_dct : bool
        perform dct transform.
    nceps : int
        number of cepstral coefficients.
    log_e : bool
        replace C0 with log energy.
    lifter : int
        postprocess the cepstrum, emphasize higher frequencies.
    deltas : bool
        concatenate first and second derivatives.
    remove_dc : bool
        remove dc offset.
    medfilt_t : int
        perform n-point median filter in the temporal domain. if 0, don't.
    medfilt_s : (int, int)
        perform (n,m)-point 2d median filter in the spectral domain.
    noise_fr : int
        perform noise subtraction based on the first n frames. if 0, don't.
    pre_emph : float
        pre-emphasis filter coefficient
    """
    def __init__(self,
                 fs=16000,
                 window_length=0.025,
                 window_shift=0.010,
                 nfft=1024,

                 scale='mel',
                 lowerf=120,
                 upperf=7000,
                 nfilt=40,
                 taper_filt=True,
                 compression='log',

                 do_dct=False,
                 nceps=13,
                 log_e=True,
                 lifter=22,

                 deltas=False,

                 remove_dc=False,
                 medfilt_t=0,
                 medfilt_s=(0,0),
                 noise_fr=0,
                 pre_emph=0.97):
        self.eps = np.finfo(np.double).eps
        self.set_scale(scale)

        self.remove_dc = remove_dc
        self.medfilt_t = medfilt_t
        self.medfilt_s = medfilt_s
        self.noise_fr = noise_fr

        if not nfilt > 0:
            raise(ValueError,
                  'Number of filters must be positive, not {0:%d}'
                  .format(nfilt))
        self.nfilt = nfilt

        self.taper_filt = taper_filt
        if upperf > fs // 2:
            raise(ValueError,
                  'Upper frequency {0:.3f} cannot exceed Nyquist {1:.3f}'
                  .format(upperf, float(fs // 2)))
        if lowerf >= upperf:
            raise(ValueError,
                  'Lower frequency {0:.3f} cannot exceed upper frequency {1:.3f}'
                  .format(lowerf, upperf))
        self.lowerf = lowerf
        self.upperf = upperf

        if not (0 <= pre_emph <= 1):
            raise ValueError('pre_emph must be between 0 and 1, not {}'
                             .format(pre_emph))
        self.pre_emph = pre_emph
        self.wlen = int(window_length * fs)
        self.win = np.hamming(self.wlen)
        self.fs = fs
        self.fshift = int(window_shift * fs)
        if not (nfft > 0 and (nfft & (nfft-1) == 0)):
            raise ValueError('NFFT must be a positive power of two, not {}'
                             .format(nfft))
        self.nfft = nfft

        compression_types = ['log', 'cubicroot', 'none']
        if not compression in compression_types:
            raise(ValueError,
                  'compression must be one of [{0:s}], not {1}'
                  .format(', '.join(compression_types), compression))
        self.compression = compression
        if self.compression == 'log':
            self.compressor = log_compression
        elif self.compression == 'cubicroot':
            self.compressor = cubicroot_compression
        else: # no compression
            self.compressor = identity_compression

        filts = np.zeros((self.nfft//2+1, self.nfilt), dtype=np.double)
        upperf = self.from_hertz(self.upperf).reshape(-1)[0]
        lowerf = self.from_hertz(self.lowerf).reshape(-1)[0]

        edges = np.round((self.nfft+1) * \
            self.to_hertz(np.linspace(lowerf,
                                      upperf,
                                      self.nfilt+2))/self.fs).astype(np.uint)
        for filt in xrange(self.nfilt):
            left = edges[filt]
            center = edges[filt+1]
            right = edges[filt+2]
            filts[left:center+1, filt] = (np.arange(left, center+1) - left) /\
                                         (center - left)
            filts[center:right+1, filt] = (right - np.arange(center, right+1)) /\
                                          (right - center)
        if self.taper_filt:
            filts = filts / filts.sum(axis=0)

        self.filters = filts

        self.lifter_c = lifter
        self.do_dct = do_dct
        self.nceps = nceps
        self.log_e = log_e
        if self.do_dct:
            self._build_dctmtx()

        self.do_deltas = deltas

    def transform(self, sig):
        """
        Convert audio signal to feature representation.

        Parameters
        ----------
        sig : ndarray (nsamples,)
            audio signal

        Returns
        -------
        frames : ndarray (nframes, nfeatures)
            feature representation

        """
        sig = sig.astype(np.double)

        # DC
        if self.remove_dc:
            sig = self.dc_filter(sig)

        # median filter
        if self.medfilt_t > 0:
            sig = ss.medfilt(sig, self.medfilt_t)

        # preemph spectral tilt
        if self.pre_emph > 0:
            sig = self.pre_emphasis(sig)

        # gain normalize:
        sig = sig / np.abs(sig).max()

        # fft
        frames = self.stft(sig)

        # power spectrum
        # frames = frames.real*frames.real + frames.imag*frames.imag
        frames = frames.real**2 / self.nfft
        if self.do_dct and self.log_e:
            # keep log energy column around for later
            log_e = np.log(frames.sum(axis=1).clip(self.eps, np.inf))

        # median filter 2d
        if self.medfilt_s[0] > 0 and self.medfilt_s[1] > 0:
            frames = ss.medfilt(frames, kernel_size=self.medfilt_s)

        # noise subtraction
        if self.noise_fr > 0:
            noise = frames[:self.noise_fr, :].mean(axis=0)
            frames /= noise

        # filter bank
        frames = np.dot(frames, self.filters).clip(self.eps, np.inf)

        # energy compression
        frames = self.compressor(frames)

        if self.do_dct:
            frames = np.dot(frames, self.dctmtx.T)
            if self.lifter_c:
                frames = self.lifter(frames)
            if self.log_e:
                frames[:,0] = log_e

        if self.do_deltas:
            frames = np.c_[frames,
                           self.deltas(frames),
                           self.deltasdeltas(frames)]

        return frames

    def dc_filter(self, sig):
        b = [1, -1]
        a = [1, -0.999]
        return ss.filtfilt(b, a, sig)

    def pre_emphasis(self, sig):
        b = [1, -self.pre_emph]
        a = 1
        zi = ss.lfilter_zi(b, a)
        return ss.lfilter(b, a, sig, zi=zi)[0]

    def stft(self, sig):
        s = np.pad(sig, (self.wlen//2, 0), 'constant')
        cols = np.ceil((s.shape[0] - self.wlen) / self.fshift + 1)
        s = np.pad(s, (0, self.wlen), 'constant')
        frames = as_strided(s, shape=(cols, self.wlen),
                            strides=(s.strides[0]*self.fshift,
                                     s.strides[0])).copy()
        return np.fft.rfft(frames*self.win, self.nfft)

    def _build_dctmtx(self):
        cols, rows = np.meshgrid(range(self.nfilt), range(self.nceps))
        dctmtx = np.sqrt(2/self.nfilt) * \
                 np.cos(np.pi*(2*cols+1)*rows/(2*self.nfilt))
        dctmtx[0,:] /= np.sqrt(2)
        self.dctmtx = dctmtx

    def lifter(self, frames):
        return frames * (1 + self.lifter_c/2. *
                         np.sin(np.pi * np.arange(self.nceps) / self.lifter_c))

    def set_scale(self, scale):
        if scale == 'hertz':
            to_hertz = np.vectorize(lambda x: x)
            from_hertz = np.vectorize(lambda x: x)
        elif scale == 'mel':
            to_hertz = np.vectorize(mel_to_hertz)
            from_hertz = np.vectorize(hertz_to_mel)
        elif scale == 'bark':
            to_hertz = np.vectorize(bark_to_hertz)
            from_hertz = np.vectorize(hertz_to_bark)
        elif scale == 'erb':
            to_hertz = np.vectorize(erb_to_hertz)
            from_hertz = np.vectorize(hertz_to_erb)
        else:
            raise ValueError('Unrecognized scale: {0}'.format(scale))

        self.scale = scale
        self.to_hertz = to_hertz
        self.from_hertz = from_hertz

    def deltas(self, X):
        nframes, nceps = X.shape
        hlen = 4
        a = np.r_[hlen:-hlen-1:-1] / 60
        g = np.r_[np.array([X[1, :] for x in range(hlen)]),
                  X,
                  np.array([X[nframes-1, :] for x in range(hlen)])]
        flt = ss.lfilter(a, 1, g.flat)
        d = flt.reshape((nframes + 8, nceps))
        return np.array(d[8:, :])

    def deltasdeltas(self, X):
        nframes, nceps = X.shape
        hlen = 4
        a = np.r_[hlen:-hlen-1:-1] / 60

        hlen2 = 1
        f = np.r_[hlen2:-hlen2-1:-1] / 2

        g = np.r_[np.array([X[1, :] for x in range(hlen+hlen2)]),
                  X,
                  np.array([X[nframes-1, :] for x in range(hlen+hlen2)])]

        flt1 = ss.lfilter(a, 1, g.flat)
        h = flt1.reshape((nframes + 10, nceps))[8:, :]

        flt2 = ss.lfilter(f, 1, h.flat)
        dd = flt2.reshape((nframes + 2, nceps))
        return dd[2:, :]


def hertz_to_mel(f):
    """
    Convert frequency in Hertz to mel.

    Parameters
    ----------
    f : float
        Frequency in Hertz.

    Returns
    -------
    float
        Frequency in mel.

    Notes
    -----

    The formulation of O'Shaughnessy [1]_ is used to calculate this function
    and its inverse `mel_to_hertz`.

    .. math:: m = 2595\log_10 (1+f/700)

    References
    ----------
    .. [1] D. O'Shaughnessy, "Speech communication: human and machine",
    Addison-Wesley, p. 150, 1987

    """
    return 2595. * np.log10(1.+f/700)


def mel_to_hertz(m):
    """
    Convert frequency in mel to Hertz.

    Parameters
    ----------
    m : float
        Frequency in mel.

    Returns
    -------
    float
        Frequency in Hertz.

    Notes
    -----
    The formulation of O'Shaughnessy [1]_ is used to calculate this function
    and its inverse `hertz_to_mel`.

    .. math:: f = 700(10^{m/2595} - 1)

    References
    ----------
    .. [1] D. O'Shaughnessy, "Speech communication: human and machine",
    Addison-Wesley, p. 150, 1987

    """
    return 700. * (np.power(10., m/2595) - 1.)

def hertz_to_bark(f):
    """
    Convert frequency in Hertz to Bark.

    Parameters
    ----------
    f : float
        Frequency in Hz.

    Returns
    -------
    float
        Frequency in Bark.

    Notes
    -----
    The formulation of Traunmueller [1]_ is used to calculate this function
    and its inverse `bark_to_hertz`.

    .. math:: z = 26.81f / (1960+f) - 0.53

    Corrections are made for the low and high frequencies.

    References
    ----------
    .. [1] H. Traunmueller, "Analytical expressions for the tonotopic sensory
    scale," J. Acoust. Soc. Am. 88(1), 1990

    """
    return 6 * np.arcsinh(f/600.)


def bark_to_hertz(z):
    """
    Convert frequency in Bark to Hertz.

    Parameters
    ----------
    z : float
        Frequency in Bark.

    Returns
    -------
    float
        Frequency in Hertz.

    Notes
    -----

    The formulation of Traunmueller [1]_ is used to calculate this function
    and its inverse `hertz_to_bark`.

    .. math:: f = \frac{52547.6}{(26.28 - z)^{1.53}}

    Corrections are made for the low and high frequencies.

    References
    ----------
    .. [1] H. Traunmueller, "Analytical expressions for the tonotopic sensory
    scale," J. Acoust. Soc. Am. 88(1), 1990

    """
    return 600 * np.sinh(z/6.)

def erb_to_hertz(e):
    """
    Convert frequency in ERB to Hertz

    Parameters
    ----------
    e : float
        Frequency in ERB

    Returns
    -------
    float

    """
    t1 = 676170.4 / (47.06538 - np.exp(0.08959494 * np.abs(e)))
    t2 = -14678.49
    return np.sign(e) * (t1 + t2)

def hertz_to_erb(f):
    """
    Convert frequency in Hertz to ERB

    Parameters
    ----------
    f : float
        Frequency in Hertz

    Returns
    -------
    float

    """
    g = np.abs(f)
    return 11.17268 * np.sign(f) * np.log(1 + 46.06538 * g / (g + 14678.49))


def log_compression(X):
    return np.log(X)

def cubicroot_compression(X):
    return X**(1./3)

def identity_compression(X):
    return X
