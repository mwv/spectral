#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()

from distutils.extension import Extension
from distutils.sysconfig import *
from distutils.util import *
from Cython.Distutils import build_ext
import numpy

py_inc = [get_python_inc()]
np_lib = os.path.dirname(numpy.__file__)
np_inc = [os.path.join(np_lib, 'core/include')]

setup(
    name='spectral',
    version='0.1.6',
    description=('Python package for extracing Mel '
                 'and MFCC features from speech.'),
    long_description=readme,
    author='Maarten Versteegh',
    author_email='maartenversteegh@gmail.com',
    url='https://github.com/mwv/spectral',
    packages=[
        'spectral',
    ],
    package_dir={'spectral': 'spectral'},
    include_package_data=True,
    install_requires=['numpy>=1.6.2', 'scipy'],
    license="BSD",
    zip_safe=False,
    keywords='spectral',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('_logspec', ['spectral/_logspec.pyx'],
                           include_dirs=py_inc + np_inc,)],
    include_dirs=[numpy.get_include(),
                  os.path.join(numpy.get_include(), 'numpy')],
    test_suite='tests',
)
