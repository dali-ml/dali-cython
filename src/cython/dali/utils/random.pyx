def randint(int low=0, int high=1):
    cdef int out
    with nogil:
        out = c_randint(low,high)
    return out

def randdouble(float low=0.0, float high=1.0):
    cdef double out
    with nogil:
        out = c_randdouble(low,high)
    return out

def reseed():
    with nogil:
        c_reseed()

def set_seed(int newseed):
    with nogil:
        c_set_seed(newseed)
