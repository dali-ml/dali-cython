# Dali in Python

[![Build Status](https://travis-ci.org/JonathanRaiman/dali-cython.svg?branch=master)](https://travis-ci.org/JonathanRaiman/dali-cython)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

This repository contains an early version of Python bindings to [Dali](https://github.com/dali-ml/Dali), the automatic differentiation framework.

With this module you can construct arbitrary mathematical models, including differentiating through control code and loops, optimize and run them using your GPU or CPU.

The bindings are light-weight, and the resulting module retains about 90%-95% of the C++ performance in most use-cases (with this performance going up for larger models and GPU intensive computations).

@author Jonathan Raiman & Szymon Sidor
@date 20th March 2016

### Installation

1. Install [Dali](https://github.com/dali-ml/Dali#installation) using `homebrew`, `yum`, or `apt-get`.

2. `pip3 install dali`

### Usage

See examples under notebooks.
