#!/usr/bin/env python

from setuptools import setup

from pandas_extras import __version__ as pe_version

with open('README.md') as desc:
    LONG_DESCRIPTION = desc.read()


setup(
    name='pandas-extras',
    version=pe_version,
    description='Extension package for the popular Pandas library',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    maintainer='Hodossy, Szabolcs',
    maintainer_email='hodossy.szabolcs@gmail.com',
    url='https://github.com/nokia/pandas-extras',
    license='BSD',
    platforms='any',
    packages=['pandas_extras'],
    keywords=['pandas', 'data analysis', 'data transformation'],
    install_requires=[
        'pandas',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)
