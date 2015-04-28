===============================
Spectral
===============================

Python package for extracing Mel and MFCC features from speech.

* Free software: GPL3 license

Features
--------

* Mel and MFCC feature extraction

Usage
-----

To extract standard-ish MFCC features with deltas from a mono wave file::

  >>> # read in the wave file
  >>> import wave
  >>> import struct
  >>> with wave.open("mywavfile.wav", 'r') as fid:
          _, _, fs, nframes, _, _ = fid.getparams()
          sig = np.array(struct.unpack_from("%dh" % nframes,
                                            fid.readframes(nframes)))
  >>> config = dict(fs=fs, scale='mel', do_dct=True, deltas=True)
  >>> extractor = spectral.Spectral(**config)
  >>> cepstra = extractor.transform(sig)

To extract 40-dimensional Bark filterbank features instead::

  >>> config = dict(fs=fs, do_dct=False, scale='bark', deltas=False)
  >>> extractor = spectral.Spectral(**config)
  >>> bark_spectrum = extractor.transform(sig)

To extract the same features but first apply some noise removal::

  >>> config = dict(fs=fs, do_dct=False, scale='bark', deltas=False,
                    remove_dc=True, medfilt_t=3, medfilt_s=(11,11), noise_fr=10)
  >>> extractor = spectral.Spectral(**config)
  >>> bark_spectrum = extractor.transform(sig)
