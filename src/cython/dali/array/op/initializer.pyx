def empty():
    return AssignableArray.wrapc(initializer_empty())

def zeros():
    return AssignableArray.wrapc(initializer_zeros())

def ones():
    return AssignableArray.wrapc(initializer_ones())

def arange(double start=0.0, double step=1.0):
    return AssignableArray.wrapc(initializer_arange(start, step))

def fill(value):
    if isinstance(value, float):
        return AssignableArray.wrapc(initializer_fill_double(value))
    elif isinstance(value, int):
        return AssignableArray.wrapc(initializer_fill_int(value))
    else:
        raise TypeError(
            "value should be of type int or float (got %r)" % (
                type(value),
            )
        )

def gaussian(double mean, double std):
    return AssignableArray.wrapc(initializer_gaussian(mean, std))

def uniform(double low, double high):
    return AssignableArray.wrapc(initializer_uniform(low, high))

def bernoulli(double prob):
    return AssignableArray.wrapc(initializer_bernoulli(prob))

def bernoulli_normalized(double prob):
    return AssignableArray.wrapc(initializer_bernoulli_normalized(prob))

def eye(double diag=1.0):
    return AssignableArray.wrapc(initializer_eye(diag))
