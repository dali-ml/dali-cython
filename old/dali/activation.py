"""
The point of this file is to make those
functions pickle friendly
"""
from dali.core import MatOps as ops

class tanh_object(object):
    def __call__(self, *args, **kwargs):
        return ops.tanh(*args, **kwargs)

class relu_object(object):
    def __call__(self, *args, **kwargs):
        return ops.relu(*args, **kwargs)

class sigmoid_object(object):
    def __call__(self, *args, **kwargs):
        return ops.sigmoid(*args, **kwargs)

class identity_object(object):
    def __call__(self, *args, **kwargs):
        assert len(args) == 1
        assert len(kwargs) == 0
        return args[0]

tanh     = tanh_object()
relu     = relu_object()
sigmoid  = sigmoid_object()
identity = identity_object()

__all__ = ["tanh", "relu", "sigmoid", "identity"]
