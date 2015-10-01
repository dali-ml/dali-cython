# Dali in Python

This repository contains an early version of Python bindings to [Dali](https://github.com/JonathanRaiman/Dali), the automatic differentiation framework.

With this module you can construct arbitrary mathematical models, including differentiating through control code and loops, optimize and run them using your GPU or CPU.

The bindings are light-weight, and the resulting module retains about 90%-95% of the C++ performance in most use-cases (with this performance going up for larger models and GPU intensive computations).

@author Jonathan Raiman & Szymon Sidor
@date 18th April 2015

### Installation

Add the following line to your .bashrc on Linux/GNU or .bash_profile on Mac OS:
```bash
export DALI_HOME="/path/to/Dali/"
```
Then you can simply install Dali by running:

```bash
python3 setup.py install
```

#### Special installation instructions for Linux

Since on Linux/GNU Dali needs to be compiled using shared libraries, we need to tell the OS where to load them from. Add the following line to your .bashrc:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DALI_HOME/build_cpu/dali
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DALI_HOME/build/dali
```

### Usage

See examples under notebooks.
