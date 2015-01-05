import numpy as np

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
    z = 26.81 * f / (1960 + f) - 0.53
    if z < 2.0:
        z = 0.3 + 0.85 * z
    elif z > 20.1:
        z = 1.22 * z - 4.422
    return z


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
    z = np.abs(z)
    if z < 2.0:
        z = (z - 0.3) / 0.85
    elif z > 20.1:
        z = (z + 4.422) / 1.22
    return 52547.6 / (26.28 - z)**1.53

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
