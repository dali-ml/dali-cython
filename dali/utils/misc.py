import dill as pickle
import inspect
import numpy as np
import types

from os import makedirs, listdir
from os.path import join, exists

import dali.core as D

class RunningAverage(object):
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.value = None

    def update(self, measurement):
        if self.value is None:
            self.value = measurement
        else:
            self.value = (self.alpha * self.value +
                         (1.0 - self.alpha) * measurement)

    def __float__(self):
        return float(self.value)


def apply_recursively_on_type(x, f, target_type, list_callback=None):
    if type(x) == target_type:
        return f(x)
    elif type(x) == list or isinstance(x, types.GeneratorType):
        ret = [ apply_recursively_on_type(el, f, target_type, list_callback) for el in x]
        if list_callback and all(type(el) == target_type for el in x):
            ret = list_callback(ret)
        return ret
    elif type(x) == dict:
        res = {}
        for k,v in x.items():
            res[k] = apply_recursively_on_type(v, f, target_type, list_callback)
        return res
    else:
        return x

def integer_ceil(a, b):
    return (a + b - 1) // b

def subsample(seq, maximum_length):
    if seq == []:
        return seq
    return seq[::integer_ceil(len(seq), maximum_length)]

def median_smoothing(signal, window=10):
    res = []
    for i in range(window, len(signal)):
        actual_window = signal[i-window:i]
        res.append(np.median(actual_window))
    return res

def pickle_from_scope(directory, variables, caller_globals=None, caller_locals=None):
    if not exists(directory):
        makedirs(directory)

    if caller_globals is None or caller_locals is None:
        stack = inspect.stack()
        if caller_globals is None:
            caller_globals = stack[1][0].f_globals
        if caller_locals is None:
            caller_locals  = stack[1][0].f_locals
        del stack

    for var in variables:
        with open(join(directory, var + ".pkz"), "wb") as f:
            value = caller_locals.get(var) or caller_globals.get(var)
            assert value is not None
            pickle.dump(value, f)

def unpickle_as_dict(directory, whitelist=None, extension='.pkz'):
    assert exists(directory)

    res = {}

    for file_name in listdir(directory):
        if file_name.endswith(extension):
            var_name = file_name[:-len(extension)]
            if whitelist is None or var_name in whitelist:
                with open(join(directory, file_name), "rb") as f:
                    res[var_name] = pickle.load(f)

    return res

def add_device_args(parser):
    parser.add_argument("--device",    type=str, default='gpu', choices=['gpu','cpu'], help="Whether model should run on GPU or CPU.")
    parser.add_argument("--gpu_id",    type=int, default=0, help="Which GPU to use (zero-indexed just like in CUDA APIs)")

def set_device_from_args(args, verbose=False):
    D.config.default_device = args.device
    if args.device == 'gpu':
        D.config.default_gpu = args.gpu_id
        if verbose:
            print("Using %s" % (D.config.gpu_id_to_name(args.gpu_id)))

__all__ = [
    "apply_recursively_on_type",
    "integer_ceil",
    "subsample",
    "median_smoothing",
    "pickle_from_scope",
    "unpickle_as_dict",
    "RunningAverage",
    "add_device_args",
    "set_device_from_args"
]
