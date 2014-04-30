import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
def pre_emphasis(np.ndarray[DTYPE_t, ndim=1] frame,
                 float prior, float alpha):
    cdef np.ndarray[DTYPE_t, ndim=1] outfr
    cdef unsigned int i, m
    m = frame.shape[0]
    outfr = np.empty(m)
    outfr[0] = frame[0] - alpha * prior
    for i in xrange(1, m):
        outfr[i] = frame[i] - alpha * frame[i-1]
    return outfr
