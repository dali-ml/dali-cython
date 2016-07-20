cdef extern from "dali/utils/random.h" namespace "utils::random" nogil:
    void c_reseed "utils::random::reseed" ()
    void c_set_seed "utils::random::set_seed" (int)

cdef extern from "dali/utils/random.h" namespace "utils" nogil:
    double c_randdouble "utils::randdouble" (double, double) except +
    int c_randint "utils::randint" (int,int) except +
