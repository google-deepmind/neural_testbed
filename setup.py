# python3
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

import imp

import setuptools

# Additional requirements for testing.
testing_require = [
    'mock',
    'pytest-xdist',
    'pytype',
]

setuptools.setup(
    name='neural_testbed',
    description=(
        'Neural testbed. '
        'A library for evaluating probabilistic inference in neural networks.'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/deepmind/neural_testbed',
    author='DeepMind',
    author_email='neural-testbed-eng+os@google.com',
    license='Apache License, Version 2.0',
    version=imp.load_source('_metadata',
                            'neural_testbed/_metadata.py').__version__,
    keywords='probabilistic-inference python machine-learning',
    packages=setuptools.find_packages(),
    install_requires=[
        'dm-haiku',
        'enn @ git+https://git@github.com/deepmind/enn',
        'absl-py',
        'numpy',
        'pandas',
        'jaxlib>=0.1.74',
        'jax',
        'ml_collections',
        'tensorflow>=2.7',
        'tensorflow-datasets',
        'tensorflow-probability',
        'chex',
        'neural-tangents',
        'dataclasses',  # Back-port for Python 3.6.
        'typing-extensions',
        'plotnine',
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
