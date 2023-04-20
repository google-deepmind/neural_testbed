# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Install script for setuptools."""

import imp  # pylint: disable=deprecated-module

import setuptools

# Additional requirements for testing.
testing_require = [
    'mock',
    'pytest-xdist',
    'pytype==2021.8.11',  # to be compatible with dm-acme
]

setuptools.setup(
    name='neural_testbed',
    description=(
        'Neural testbed. '
        'A library for evaluating probabilistic inference in neural networks.'
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/deepmind/neural_testbed',
    author='DeepMind',
    author_email='neural-testbed-eng+os@google.com',
    license='Apache License, Version 2.0',
    version=imp.load_source(
        '_metadata', 'neural_testbed/_metadata.py'
    ).__version__,
    keywords='probabilistic-inference python machine-learning',
    packages=setuptools.find_packages(),
    install_requires=[
        'absl-py',
        'chex',
        'dm-haiku',
        'enn @ git+https://git@github.com/deepmind/enn',
        'jax',
        'jaxlib',
        'ml_collections',
        'neural-tangents',
        'numpy',
        'pandas',
        'plotnine',
        'tensorflow==2.8.0',  # to be compatible with dm-acme
        'tensorflow-datasets==4.6.0',  # to be compatible with dm-acme
        'tensorflow_probability==0.15.0',  # to be compatible with dm-acme
        'typing-extensions',
        'protobuf==3.20.0',  # to avoid Typeerror: descriptors cannot not be 
        # created directly
    ],
    extras_require={
        'testing': testing_require,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
