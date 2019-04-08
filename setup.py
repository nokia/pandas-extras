#!/usr/bin/env python

from setuptools import setup

with open('README.md') as desc:
    LONG_DESCRIPTION = desc.read()


setup(
    name='pandas-extras',
    version='0.0.1',
    description='Extension package for the popular Pandas library',
    long_description=LONG_DESCRIPTION,
    maintainer='Hodossy, Szabolcs',
    maintainer_email='hodossy.szabolcs@gmail.com',
    url='https://github.com/hodossy/pandas-extras',
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.5',
)
