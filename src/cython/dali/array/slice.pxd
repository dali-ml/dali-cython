from dali.array.array   cimport *
from dali.tensor.tensor cimport *

cdef extern from "dali/utils/optional.h":
    cdef cppclass OptionalInt "std::experimental::optional<int>":
        OptionalInt()  except +
        OptionalInt(int x)  except +

cdef extern from "dali/array/slice.h":
    cdef cppclass CBroadcast "Broadcast":
        pass

    cdef cppclass CSlice "Slice":
        CSlice()  except +
        CSlice(const OptionalInt& start, const OptionalInt& end, const OptionalInt& step)  except +

    cdef cppclass CSlicingInProgressArray "SlicingInProgress<Array>":
        CArray toarray "operator Array"()  except +
        CSlicingInProgressArray operator_bracket "operator[]"(const int&)  except +
        CSlicingInProgressArray operator_bracket "operator[]"(const CSlice&)  except +
        CSlicingInProgressArray operator_bracket "operator[]"(const CBroadcast&)  except +

    cdef cppclass CSlicingInProgressTensor "SlicingInProgress<Tensor>":
        CTensor totensor "operator Tensor"()  except +
        CSlicingInProgressTensor operator_bracket "operator[]"(const int&)  except +
        CSlicingInProgressTensor operator_bracket "operator[]"(const CSlice&)  except +
        CSlicingInProgressTensor operator_bracket "operator[]"(const CBroadcast&)  except +

cdef inline CSlice parse_slice(slice s) except +:
    cdef OptionalInt start
    cdef OptionalInt end
    cdef OptionalInt step

    start = OptionalInt(s.start) if s.start is not None else OptionalInt()
    end   = OptionalInt(s.stop)  if s.stop  is not None else OptionalInt()
    step  = OptionalInt(s.step)  if s.step  is not None else OptionalInt()

    return CSlice(start, end, step)
