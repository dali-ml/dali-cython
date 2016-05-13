import distutils.ccompiler
import distutils.sysconfig
import numpy as np
import preprocessor
import subprocess

from Cython.Distutils.extension import Extension
from Cython.Distutils           import build_ext
from distutils.core             import setup
from distutils.command          import build as build_module, clean as clean_module
from distutils.spawn            import find_executable
from os.path                    import join, dirname, realpath, exists, getmtime, relpath
from os                         import environ, walk, makedirs
from sys                        import platform, exit


from tempfile import TemporaryDirectory

SCRIPT_DIR = dirname(realpath(__file__))
DALI_CORE_DIR    = join(SCRIPT_DIR, "cython", "dali", "core")
DALI_CORE_MODULE = "dali.core"

################################################################################
##                               TOOLS                                        ##
################################################################################

def find_extension_files(path, extension):
    """Recursively find files with specific extension in a directory"""
    for relative_path, dirs, files in walk(path):
        for fname in files:
            if fname.endswith(extension):
                yield join(path, relative_path, fname)

def execute_bash(command, *args, **kwargs):
    """Executes bash command, prints output and throws an exception on failure."""
    #print(subprocess.check_output(command.split(' '), shell=True))
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True,
                               *args, **kwargs)
    process.wait()
    return str(process.stdout.read()), process.returncode

################################################################################
##                 STEALING LINKING ARGS FROM CMAKE                           ##
################################################################################

def cmake_robbery(varnames, fake_executable="dummy"):
    """Capture Cmake environment variables by running `find_package(dali)`"""
    varstealers = []
    magic_command = "CYTHON_DALI_BEGIN_VARIABLE_STEALING"
    varstealers.append("message(STATUS \"%s\")" % (magic_command,))
    for varname in varnames:
        varstealers.append("message(STATUS  \"CYTHON_DALI_%s: ${%s}\")" % (varname, varname,))
    varstealers = "\n".join(varstealers) + "\n"

    with TemporaryDirectory() as temp_dir:
        with open(join(temp_dir, "source.cpp"), "wt") as source_cpp:
            source_cpp.write("int main() {};\n")
        with open(join(temp_dir, "CMakeLists.txt"), "wt") as cmake_conf:
            cmake_conf.write("""
                cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
                project("dali-cython")
                find_package(Dali REQUIRED) # find Dali.
                add_executable(%s source.cpp)
                target_link_libraries(%s ${DALI_AND_DEPS_LIBRARIES})
            """ % (fake_executable, fake_executable,) + varstealers)

        cmake_subdirectory = fake_executable + ".dir"
        cmake_stdout, cmake_status = execute_bash(["cmake", "."], cwd=temp_dir)
        if cmake_status != 0:
            print("HORRIBLE CMAKE ERROR.")
            print('*' * 79)
            print(cmake_stdout)
            print('*' * 79)
            exit(1)
        # capture the link arguments
        with open(join(temp_dir, "CMakeFiles", cmake_subdirectory, "link.txt"), "rt") as f:
            linking_command = f.read()

    linking_command = linking_command.replace("-o %s" % (fake_executable,), " ")
    linking_args = linking_command.split(" ", 1)[1].strip().split()
    linking_args = [arg for arg in linking_args if cmake_subdirectory not in arg]
    outvars = {}
    outvars["LINK_ARGS"] = linking_args

    # slice output after the magic command and retrieve these variables
    # from the CMake environment
    idx = cmake_stdout.find(magic_command) + len(magic_command) + 1
    lines = cmake_stdout[idx:].split("\n")[:len(varnames)]

    for varname, line in zip(varnames, lines):
        assert(varname in line)
        _, value = line.split(":", 1)
        outvars[varname] = value.strip().split(";")
    return outvars

# cmake environment variables
robbed = cmake_robbery(["DALI_AND_DEPS_INCLUDE_DIRS"])

################################################################################
##                 AUTODETECTING COMPILER VERSION                             ##
################################################################################

class Version(tuple):
    @staticmethod
    def from_string(version_str):
        return Version([int(n) for n in version_str.split('.')])

    def __str__(self):
        return '.'.join([str(n) for n in self])

def detect_compiler(possible_commands, version_extractor, min_version):
    good_executable = None
    good_version    = None
    for command in possible_commands:
        executable = find_executable(command)
        if executable is None: continue
        version = version_extractor(executable)
        if version is None: continue
        if version >= min_version:
            good_executable = executable
            good_version    = version
            break
    return good_executable, good_version

def obtain_gxx_version(gcc_executable):
    try:
        gcc_version, status = execute_bash([gcc_executable, '-dumpversion'])
        assert status == 0
        return Version.from_string(gcc_version)
    except Exception:
        return None

GXX_VERSION_ERROR = \
"""Minimum required version of gcc/g++ must is %s.

We strive to cover all the cases for automatic compiler detection,
however if we failed to detect yours please kindly report it on github.

You can explicitly specify an executables by running:

    CC=/path/to/my/gcc CXX=/path/to/my/g++ python3 setup.py ...

"""

# set the compiler unless explicitly specified.
if platform == 'linux':
    for env_var, possible_commands, min_version in [
                ('CC',  ['gcc', 'gcc4.9', 'gcc-4.9'], Version((4, 9))),
                ('CXX', ['g++', 'g++4.9', 'g++-4.9'], Version((4, 9))),
            ]:
        if env_var not in environ:
            gxx_executable, gxx_version = detect_compiler(possible_commands, obtain_gxx_version, min_version)
            if gxx_executable is None:
                print(GXX_VERSION_ERROR % (str(min_version),))
                exit(2)
            else:
                print('Autodetected %s executable %s, version: %s' % (env_var, gxx_executable, str(gxx_version)))
                environ[env_var] = gxx_executable
else:
    if "CC" not in environ:
        environ["CC"]  = "clang"
    if "CXX" not in environ:
        environ["CXX"] = "clang++"

################################################################################
##                      TAKING OUT THE TRASH                                  ##
################################################################################


# Make a `cleanall` rule to get rid of intermediate and library files
class clean(clean_module.clean):
    def run(self):
        print("Cleaning up cython files...")
        # Just in case the build directory was created by accident,
        # note that shell=True should be OK here because the command is constant.
        for place in ["build", "cython/dali/core.c", "cython/dali/core.cpp", "dali/*.so", "MANIFEST.in"]:
            subprocess.Popen("rm -rf %s" % (place,), shell=True, executable="/bin/bash", cwd=SCRIPT_DIR)

compiler = distutils.ccompiler.new_compiler()
distutils.sysconfig.customize_compiler(compiler)
BLACKLISTED_COMPILER_SO = ['-Wp,-D_FORTIFY_SOURCE=2']
build_ext.compiler = compiler

ext_modules = [Extension(
    name=DALI_CORE_MODULE,
    sources=[join(SCRIPT_DIR, "cython", "dali", "core.pyx")] + list(find_extension_files(DALI_CORE_DIR, ".cpp")),
    library_dirs=[],
    language='c++',
    extra_compile_args=['-std=c++11'],
    extra_link_args=robbed["LINK_ARGS"],
    libraries=[],
    extra_objects=[],
    include_dirs=[np.get_include()] + robbed["DALI_AND_DEPS_INCLUDE_DIRS"]
)]


################################################################################
##       PREPROCSSOR - HOW TO SHRINK DALI CYTHON CODE THREEFOLD               ##
################################################################################


def run_preprocessor():
    """
    Generate python files using a file prepocessor (essentially macros
    that generate multiple versions of the code for each dtype supported
    by a Dali operation)
    """
    EXTENSION = ".pre"
    for py_processor_file in find_extension_files(SCRIPT_DIR, EXTENSION):
        output_file = py_processor_file[:-len(EXTENSION)]

        if not exists(output_file) or \
                getmtime(py_processor_file) > getmtime(output_file) or \
                getmtime(join(SCRIPT_DIR, "preprocessor_utils.py")) > getmtime(output_file):
            print('Preprocessing %s' % (py_processor_file,))
            with open(output_file, "wt") as f:
                f.write(preprocessor.process_file(py_processor_file, prefix='pyp', suffix='ypy'))



################################################################################
##                 POSSIBLY NO LONGER NEEDED                                  ##
################################################################################


# We need to remove some compiler flags, to make sure
# the code can compile on Fedora (to be honest it seems
# to be a bug in Fedora's distrubtion of Clang).
# Nevertheless this little madness below is to change
# default compiler flags used by Cython.
# If you know a better way call me immediately day
# or night at 4getszymo4. Thank you!
class nonbroken_build_ext(build_ext):
    def build_extensions(self, *args, **kwargs):
        run_preprocessor()
        new_compiler_so = []
        for arg in self.compiler.compiler_so:
            if arg not in BLACKLISTED_COMPILER_SO:
                new_compiler_so.append(arg)
        self.compiler.compiler_so = new_compiler_so
        super(nonbroken_build_ext, self).build_extensions(*args, **kwargs)


################################################################################
##                 FIND ALL THE FILES AND CONFIGURE SETUP                     ##
################################################################################

# generate manifest.in
pre_files = list(find_extension_files(SCRIPT_DIR, ".pre"))
pyx_files = list(find_extension_files(SCRIPT_DIR, ".pyx"))
# check that this file was not auto-generated
pyx_files = [fname for fname in pyx_files if fname + ".pre" not in pre_files]
cpp_files = list(find_extension_files(DALI_CORE_DIR, ".cpp"))
header_files = list(find_extension_files(DALI_CORE_DIR, ".h"))
pxd_files = (
    list(find_extension_files(join(SCRIPT_DIR, "libcpp11"), ".pxd")) +
    list(find_extension_files(join(SCRIPT_DIR, "modern_numpy"), ".pxd"))
)

with open(join(SCRIPT_DIR, "MANIFEST.in"), "wt") as manifest_in:
    for fname in pre_files + pyx_files + cpp_files + header_files + pxd_files + [join(SCRIPT_DIR, "preprocessor_utils.py")]:
        manifest_in.write("include %s\n" % (relpath(fname, SCRIPT_DIR)))

setup(
  name="dali",
  version='1.0.9',
  cmdclass={"build_ext": nonbroken_build_ext, 'clean': clean},
  ext_modules=ext_modules,
  description="Buttery smooth automatic differentiation using Dali.",
  author="Jonathan Raiman, Szymon Sidor",
  author_email="jonathanraiman at gmail dot com",
  install_requires=[
    'preprocessor',
    'numpy',
    'dill'
  ],
  packages=[
    "dali",
    "dali.utils",
    "dali.data",
    "dali.models",
  ]
)
