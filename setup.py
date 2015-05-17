import distutils.ccompiler
import distutils.sysconfig

from os.path import join
from os import environ

from distutils.core import setup
from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext

modname     = "test_dali"

environ["CC"] = "clang++"
environ["CXX"] = "clang++"

DALI_DIR = environ["DALI_HOME"]

compiler = distutils.ccompiler.new_compiler()

distutils.sysconfig.customize_compiler(compiler)

BLACKLISTED_COMPILER_SO = [
  '-Wp,-D_FORTIFY_SOURCE=2'
]

build_ext.compiler = compiler

ext_modules = [Extension(
    name=modname,
    sources=[
      "test_dali.pyx"],
    language='c++',
    extra_compile_args=[
        '-std=c++11',
        '-fPIC'
    ],
    libraries=[
        "protobuf",
        "sqlite3",
    ],
    extra_objects=[
        join(DALI_DIR, "build/dali/libdali.a"),
        join(DALI_DIR, "build/protobuf/libproto.a"),
        join(DALI_DIR, "build/third_party/SQLiteCpp/libSQLiteCpp.a"),
        join(DALI_DIR, "build/third_party/json11/libjson11.a")
    ],
    include_dirs=[
        DALI_DIR,
        join(DALI_DIR, "third_party/SQLiteCpp/include"),
        join(DALI_DIR, "third_party/json11"),
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]
)]

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
