from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from Cython.Build.Dependencies import create_extension_list

modname     = "test_dali"
from os.path import join
from os import environ

DALI_DIR = environ["DALI_HOME"]

ext_modules = [Extension(
    name=modname,
    sources=[
      "test_dali.pyx"],
    language='c++',
    extra_compile_args=[
        '-stdlib=libc++',
        '-std=c++11'],
    libraries=["protobuf"],
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
        "/usr/include/eigen3"
      ]
     )]

# ext_options = {
#   "language": "c++",
#   "extra_compile_args": [
#     '-stdlib=libc++',
#     '-std=c++11'],
#   "libraries": ["protobuf"],
#   "extra_objects":[
#     join(DALI_DIR, "build/dali/libdali.a"),
#     join(DALI_DIR, "build/protobuf/libproto.a"),
#     join(DALI_DIR, "build/third_party/SQLiteCpp/libSQLiteCpp.a")
#   ],
#   "include_dirs":[
#     DALI_DIR,
#     join(DALI_DIR, "third_party/SQLiteCpp/include"),
#     "/usr/local/include/eigen3"
#   ]
# }

# modules, depends = create_extension_list(["test_dali.pyx", "*/*/*.pyx"])

# for module in modules:
#   for key, value in ext_options.items():
#     setattr(module, key, value)

setup(
  name = modname,
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)
