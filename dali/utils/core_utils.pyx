cdef extern from "dali/utils/core_utils.h" namespace "utils":
    string cpp_trim "utils::trim" (string)
    string cpp_ltrim "utils::ltrim" (string)
    string cpp_rtrim "utils::rtrim" (string)

class utils:
    @staticmethod
    def trim(s):
        """Trim a string, remove whitespace on either side"""
        cdef string s_norm = normalize_s(s)
        return cpp_trim(s_norm)

    @staticmethod
    def rtrim(s):
        """Trim right-side of a string."""
        cdef string s_norm = normalize_s(s)
        return cpp_rtrim(s_norm)

    @staticmethod
    def ltrim(s):
        """Trim left-side of a string."""
        cdef string s_norm = normalize_s(s)
        return cpp_ltrim(s_norm)
