from libcpp.string   cimport string
from .cython_utils cimport normalize_s

cdef extern from "utils.h" nogil:
    void print_str(string)


def print_from_cpp(s):
    """Print a string using iostream"""
    cdef string s_norm = normalize_s(s)
    print_str(s_norm)
