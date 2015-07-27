import sys
import subprocess
import distutils.ccompiler
import distutils.sysconfig

from os.path import join, dirname, realpath
from os      import environ
import numpy as np

from distutils.core import setup
from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext

modname     = "test_dali"
environ["CC"] = "clang++"
environ["CXX"] = "clang++"

DALI_DIR   = environ["DALI_HOME"]
SCRIPT_DIR = dirname(realpath(__file__))

args = sys.argv[1:]

# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print("Cleaning up cython files...")
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash", cwd = SCRIPT_DIR)
    subprocess.Popen("rm -rf test_dali.c", shell=True, executable="/bin/bash",   cwd = SCRIPT_DIR)
    subprocess.Popen("rm -rf test_dali.cpp", shell=True, executable="/bin/bash",   cwd = SCRIPT_DIR)
    subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash",  cwd = SCRIPT_DIR)

    # Now do a normal clean
    sys.argv[sys.argv.index("cleanall")] = "clean"

use_cuda = False

if "cuda" in args:
    use_cuda = True
    sys.argv.pop(sys.argv.index("cuda"))

CUDA_INCLUDE_DIRS = []
CUDA_LIBRARIES    = []
CUDA_MACROS       = [("MSHADOW_USE_CUDA", "0")]
CUDA_EXTRA_COMPILE_ARGS = []
CUDA_LIBRARY_DIRS = []

if use_cuda:
    DALI_BUILD_DIR    = join(DALI_DIR, "build")
    CUDA_INCLUDE_DIRS = ["/usr/local/cuda/include"]
    CUDA_MACROS       = [("MSHADOW_USE_CUDA", "1")]
    DALI_OBJECTS      = [join(DALI_BUILD_DIR, "dali", "libdali.a"),
                         join(DALI_BUILD_DIR, "dali", "libdali_cuda.a")]
    CUDA_LIBRARIES    = ["cudart", "cublas", "curand"]

    CUDA_LIBRARY_DIRS = ["/usr/local/cuda/lib/"]

    CUDA_EXTRA_COMPILE_ARGS = []

else:
    # use build_cpu and don't include cuda headers
    DALI_BUILD_DIR    = join(DALI_DIR, "build_cpu")
    DALI_OBJECTS      = [join(DALI_BUILD_DIR, "dali", "libdali.a")]



compiler = distutils.ccompiler.new_compiler()
distutils.sysconfig.customize_compiler(compiler)
BLACKLISTED_COMPILER_SO = [
  '-Wp,-D_FORTIFY_SOURCE=2'
]
build_ext.compiler = compiler
ext_modules = [Extension(
    name=modname,
    sources=[
        "test_dali.pyx",
        join(SCRIPT_DIR, "dali", "tensor", "python_tape.cpp"),
        join(SCRIPT_DIR, "dali", "tensor", "matrix_initializations.cpp")
    ],
    library_dirs=CUDA_LIBRARY_DIRS,
    language='c++',
    extra_compile_args=[
        '-std=c++11',
    ] + CUDA_EXTRA_COMPILE_ARGS,
    define_macros = [('MSHADOW_USE_CBLAS','1'),
                     ('MSHADOW_USE_MKL',  '0'),
                     ('DALI_USE_VISUALIZER', '1'),
                     ('DALI_DATA_DIR', join(DALI_DIR, "data"))] + CUDA_MACROS,
    libraries=[
        "protobuf",
        "sqlite3",
        "cblas"
    ] + CUDA_LIBRARIES,
    extra_objects=DALI_OBJECTS + [
        join(DALI_BUILD_DIR, "protobuf", "libproto.a"),
        join(DALI_BUILD_DIR, "third_party", "SQLiteCpp", "libSQLiteCpp.a"),
        join(DALI_BUILD_DIR, "third_party", "json11", "libjson11.a"),
        "/usr/local/lib/libgflags.a",
    ],
    include_dirs=[
        DALI_DIR,
        join(DALI_DIR, "third_party/SQLiteCpp/include"),
        join(DALI_DIR, "third_party/json11"),
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
        join(DALI_DIR, "third_party/libcuckoo/src")
    ] + CUDA_INCLUDE_DIRS
      + [np.get_include()]
)]

# We need to remove some compiler flags, to make sure
# the code can compile on Fedora (to be honest it seems
# to be a bug in Fedora's distrubtion of Clang).
# Nevertheless this little madness below is to change
# default compiler flags used by Cython.
# If you know a better way call me immediately day
# or night at 4getszymo4. Thank you!
class nonbroken_build_ext(build_ext):
    def build_extensions(self, *args, **kwargs):
        new_compiler_so = []
        for arg in self.compiler.compiler_so:
            if arg not in BLACKLISTED_COMPILER_SO:
                new_compiler_so.append(arg)
        self.compiler.compiler_so = new_compiler_so
        super(nonbroken_build_ext, self).build_extensions(*args, **kwargs)
setup(
  name = modname,
  cmdclass = {"build_ext": nonbroken_build_ext},
  ext_modules = ext_modules
)
