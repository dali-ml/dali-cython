import distutils.ccompiler
import distutils.sysconfig
import numpy as np
import os
import re
import subprocess
import sys
import tempfile
import subprocess

from Cython.Distutils           import build_ext
from Cython.Distutils.extension import Extension
from distutils.command          import build as build_module, clean as clean_module
from distutils.core             import setup
from distutils.spawn            import find_executable
from os                         import environ, walk, makedirs
from os.path                    import join, dirname, realpath, exists, getmtime, relpath
from sys                        import platform, exit


from tempfile import TemporaryDirectory

SCRIPT_DIR       = dirname(realpath(__file__))
DALI_SOURCE_DIR  = join(SCRIPT_DIR, "src")
DALI_MODULE_NAME = "dali"

################################################################################
##                               TOOLS                                        ##
################################################################################

def find_files_by_suffix(path, suffix):
    """Recursively find files with specific suffix in a directory"""
    for relative_path, dirs, files in walk(path):
        for fname in files:
            if fname.endswith(suffix):
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


FIND_CYTHON_CPP_INCLUDES = re.compile('^[^#]+extern\W+from\W+"(?P<path>[^"]+)"')

def extract_cython_cpp_include_paths(file_path):
    with open(file_path, "rt") as f:
        for line in f:
            res = FIND_CYTHON_CPP_INCLUDES.search(line)
            if res is not None:
                yield res.groupdict()['path']

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

    if 'DALI_HOME' in os.environ and os.environ['DALI_HOME'] != '':
        if not os.path.exists(join(os.environ['DALI_HOME'], 'cmake', 'DaliConfig.cmake')):
            raise Exception(("DALI_HOME environment variable set to {}, but "
                             "no cmake/DaliConfig.cmake found at that location.").format(os.environ["DALI_HOME"]))

    with TemporaryDirectory() as temp_dir:
        with open(join(temp_dir, "source.cpp"), "wt") as source_cpp:
            source_cpp.write("int main() {};\n")
        with open(join(temp_dir, "CMakeLists.txt"), "wt") as cmake_conf:
            cmake_conf.write("""
                cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
                project("dali-cython")

                if (DEFINED ENV{DALI_HOME} AND NOT "$ENV{DALI_HOME}" STREQUAL "")
                    set(Dali_DIR $ENV{DALI_HOME}/cmake)
                endif()

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

def get_macro_definitions(varnames, config_path):
    with open(config_path, "rt") as fin:
        code = fin.read()
    with tempfile.NamedTemporaryFile(suffix=".cpp") as c_file:
        fname = c_file.name
        with open(fname, "wt") as fout:
            fout.write(code)
            for varname in varnames:
                fout.write("#ifdef " + varname + "\nDECLARE \"" + varname + "\" " + varname + "\n")
                fout.write("#else\nDECLARE \"" + varname + "\" undefined\n#endif\n")
        macro_extraction = subprocess.check_output(["cpp", fname], universal_newlines=True)#, stderr=subprocess.DEVNULL)
        found_values = {}
        for line in macro_extraction.split("\n"):
            if not line.startswith("DECLARE "):
                continue
            name_value = line[len("DECLARE "):].strip().split(" ")
            name = name_value[0].strip('"')
            if len(name_value) == 1:
                found_values[name] = True
            else:
                values = name_value[1:]
                if len(values) > 1:
                    found_values[name] = " ".join(value)
                else:
                    if values[0] == "undefined":
                        found_values[name] = False
                    else:
                        try:
                            found_values[name] = int(values[0])
                        except:
                            found_values[name] = values[0]
        return found_values

varnames = [
    "DALI_USE_CUDA",
    "DALI_USE_CUDNN",
    "DALI_USE_VISUALIZER",
    "DALI_MAX_STRIDED_DIMENSION",
    "DALI_MAX_GPU_DEVICES",
    "MSHADOW_USE_CUDA",
    "MSHADOW_USE_CBLAS",
    "MSHADOW_USE_MKL",
    "MSHADOW_FORCE_STREAM",
    "WITH_PRETTY_STACKTRACES",
    "DALI_APPLE_STACKTRACES"
]

location_of_dali_config = None
for name in robbed["DALI_AND_DEPS_INCLUDE_DIRS"]:
    if os.path.exists(join(name, "dali", "config.h")):
        location_of_dali_config = join(name, "dali", "config.h")
        break

if location_of_dali_config is not None:
    macro_values = get_macro_definitions(varnames, location_of_dali_config)
else:
    raise Exception(
        "could not find dali/config.h in any of the install locations for Dali.\n"
        " Please ensure that the headers for Dali are present on the computer to install dali."
    )

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
##                      FIND EXTENSION MODULES                                ##
################################################################################

def path_to_module_name(path):
    BASE_DIRS = ["python", "cython"]
    relative_path = os.path.relpath(path, join(DALI_SOURCE_DIR))
    path_no_ext, _ = os.path.splitext(relative_path)
    for base_dir in BASE_DIRS:
        if path_no_ext.startswith(base_dir):
            return path_no_ext.lstrip(base_dir + os.sep).replace(os.sep, '.')
    raise Exception("Cannot convert path %r to module name" % (relative_path,))


compiler = distutils.ccompiler.new_compiler()
distutils.sysconfig.customize_compiler(compiler)
BLACKLISTED_COMPILER_SO = ['-Wp,-D_FORTIFY_SOURCE=2']
build_ext.compiler = compiler

ext_modules = []
include_dirs = [np.get_include()] + robbed["DALI_AND_DEPS_INCLUDE_DIRS"] + [join(DALI_SOURCE_DIR, "cpp")]
macro_values_as_tuples = sorted(list(macro_values.items()))

config_file_path = join(DALI_SOURCE_DIR, "cython", "dali", "config.pxi")
with open(config_file_path, "wt") as fout:
    fout.write("# config.pxi is auto-generated by setup.py. Do not modify me\n")
    for key, value in macro_values_as_tuples:
        fout.write("DEF %s = %r\n" % (key, value))

# print(macro_values_as_tuples)
for pyx_file in find_files_by_suffix(join(DALI_SOURCE_DIR, "cython"), ".pyx"):
    extra_cpp_sources = []

    # pxd files are like header files for pyx files
    # and they can also have relevant includes.
    relevant_files = [pyx_file]
    pxd_file = pyx_file.rstrip("pyx") + "pxd"
    if os.path.exists(pxd_file):
        relevant_files.append(pxd_file)

    # find all the cpp files referenced from pyx files
    # and if some exist in src/cpp folder, compile them
    # as well
    for cpy_file in relevant_files:
        for header_path in extract_cython_cpp_include_paths(cpy_file):
            hypothetical_source_path = header_path.rstrip('.h') + '.cpp'
            hypothetical_source_full_path = join(DALI_SOURCE_DIR, 'cpp', hypothetical_source_path)
            if os.path.exists(hypothetical_source_full_path):
                extra_cpp_sources.append(hypothetical_source_full_path)
    ext_modules.append(Extension(
        name=path_to_module_name(pyx_file),
        sources=[pyx_file] + extra_cpp_sources,
        library_dirs=[],
        language='c++',
        extra_compile_args=['-std=c++11', '-Wno-unused-function', '-Wno-unused-local-typedef', '-Wno-undefined-bool-conversion'],
        extra_link_args=robbed["LINK_ARGS"],
        libraries=[],
        extra_objects=[],
        include_dirs=include_dirs
    ))

################################################################################
##                      FIND PYTHON PACKAGES                                  ##
################################################################################

py_packages = []
for file in find_files_by_suffix(join(DALI_SOURCE_DIR, "python"), "__init__.py"):
    module_path = dirname(file)
    py_packages.append(path_to_module_name(module_path))

################################################################################
##              BUILD COMMAND WITH EXTRA WORK WHEN DONE                       ##
################################################################################

def symlink_built_package():
    build_dir_contents = os.listdir(join(SCRIPT_DIR, "build"))
    lib_dot_fnames = []
    for name in build_dir_contents:
        if name.startswith("lib."):
            lib_dot_fnames.append(join(SCRIPT_DIR, "build", name))
    # get latest lib. file created and symlink it to the project
    # directory for easier testing
    lib_dot_fnames = sorted(
        lib_dot_fnames,
        key=lambda name: os.stat(name).st_mtime,
        reverse=True
    )
    if len(lib_dot_fnames) == 0:
        return

    most_recent_name = join(lib_dot_fnames[0], DALI_MODULE_NAME)
    symlink_name = join(SCRIPT_DIR, DALI_MODULE_NAME)

    if os.path.lexists(symlink_name):
        if os.path.islink(symlink_name):
            os.remove(symlink_name)
        else:
            print(
                ("non symlink file with name %r found in project directory."
                " Please remove to create a symlink on build") % (
                    symlink_name,
                )
            )
            return

    os.symlink(
        most_recent_name,
        symlink_name,
        target_is_directory=True
    )
    print("Created symlink pointing to %r from %r" % (
        most_recent_name,
        join(SCRIPT_DIR, DALI_MODULE_NAME))
    )

class build_with_posthooks(build_module.build):
    def run(self):
        build_module.build.run(self)
        symlink_built_package()




# Make a `cleanall` rule to get rid of intermediate and library files
class clean_with_posthooks(clean_module.clean):
    def run(self):
        clean_module.clean.run(self)

        # remove cython generated sources
        for file_path in find_files_by_suffix(join(DALI_SOURCE_DIR, 'cython'), '.cpp'):
            os.remove(file_path)
        if os.path.exists(config_file_path):
            os.remove(config_file_path)

################################################################################
##                 FIND ALL THE FILES AND CONFIGURE SETUP                     ##
################################################################################

# # generate manifest.in
# pre_files = list(find_files_by_suffix(SCRIPT_DIR, ".pre"))
# pyx_files = list(find_files_by_suffix(SCRIPT_DIR, ".pyx"))
# # check that this file was not auto-generated
# pyx_files = [fname for fname in pyx_files if fname + ".pre" not in pre_files]
# cpp_files = list(find_files_by_suffix(DALI_CORE_DIR, ".cpp"))
# header_files = list(find_files_by_suffix(DALI_CORE_DIR, ".h"))
# pxd_files = (
#     list(find_files_by_suffix(join(SCRIPT_DIR, "libcpp11"), ".pxd")) +
#     list(find_files_by_suffix(join(SCRIPT_DIR, "modern_numpy"), ".pxd"))
# )

# with open(join(SCRIPT_DIR, "MANIFEST.in"), "wt") as manifest_in:
#     for fname in pre_files + pyx_files + cpp_files + header_files + pxd_files + [join(SCRIPT_DIR, "preprocessor_utils.py")]:
#         manifest_in.write("include %s\n" % (relpath(fname, SCRIPT_DIR)))

setup(
  name="dali",
  version='1.1.0',
  cmdclass={"build": build_with_posthooks, 'build_ext': build_ext, 'clean': clean_with_posthooks},
  ext_modules=ext_modules,
  description="Buttery smooth automatic differentiation.",
  author="Jonathan Raiman, Szymon Sidor",
  author_email="jonathanraiman at gmail dot com",
  package_dir={'': join(DALI_SOURCE_DIR, 'python')},
  install_requires=[
    'numpy',
    'dill'
  ],
  packages=py_packages
)
