from dali.core import MatOps as ops

class tanh(object):
    def __call__(self, *args, **kwargs):
        return ops.tanh(*args, **kwargs)

class relu(object):
    def __call__(self, *args, **kwargs):
        return ops.relu(*args, **kwargs)

class sigmoid(object):
    def __call__(self, *args, **kwargs):
        return ops.sigmoid(*args, **kwargs)
