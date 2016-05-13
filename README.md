# Dali in Python

[![PyPI version](https://badge.fury.io/py/dali.svg)](https://badge.fury.io/py/dali)
[![Build Status](https://travis-ci.org/dali-ml/dali-cython.svg?branch=master)](https://travis-ci.org/dali-ml/dali-cython)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

This repository contains an early version of Python bindings to [Dali](https://github.com/dali-ml/Dali), the automatic differentiation framework.

With this module you can construct arbitrary mathematical models, including differentiating through control code and loops, optimize and run them using your GPU or CPU.

The bindings are light-weight, and the resulting module retains about 90%-95% of the C++ performance in most use-cases (with this performance going up for larger models and GPU intensive computations).

[![Jonathan Raiman, author](https://img.shields.io/badge/Author-Jonathan%20Raiman%20-blue.svg)](https://github.com/JonathanRaiman/) [![Szymon Sidor, author](https://img.shields.io/badge/Author-Szymon%20Sidor%20-blue.svg)](https://github.com/nivwusquorum)

### Installation

1. Install [Dali](https://github.com/dali-ml/Dali#installation) using `homebrew`, `yum`, or `apt-get`.

2. `pip3 install dali`

### Usage

See examples under notebooks.
