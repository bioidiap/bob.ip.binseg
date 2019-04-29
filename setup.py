#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements, find_packages
install_requires = load_requirements()


setup(

    name='bob.ip.binseg',
    version=open("version.txt").read().rstrip(),
    description='Binary Segmentation Benchmark Package for Bob',

    url='https://gitlab.idiap.ch/bob/bob.ip.binseg',
    license='GPLv3',

    # there may be multiple authors (separate entries by comma)
    author='Tim Laibacher',
    author_email='tim.laibacher@idiap.ch',

    # there may be a maintainer apart from the author - you decide
    #maintainer='?'
    #maintainer_email='email@example.com'

    # you may add more keywords separating those by commas (a, b, c, ...)
    keywords = "bob",

    long_description=open('README.rst').read(),

    # leave this here, it is pretty standard
    packages=find_packages(),
    include_package_data=True,
    zip_safe = False,

    install_requires=install_requires,

  entry_points={

        # main entry for bob hed cli
        'bob.cli': [
            'binseg = bob.ip.binseg.script.binseg:binseg',
        ],

         #bob hed sub-commands
        'bob.ip.binseg.cli': [
          'train = bob.ip.binseg.script.binseg:train',
          'test = bob.ip.binseg.script.binseg:test',
          'compare =  bob.bin.binseg.script.binseg:compare',
          'testcheckpoints = bob.ip.binseg.script.binseg:testcheckpoints',
        ],

         #bob hed train configurations
        'bob.ip.binseg.config': [
          'DRIU = bob.ip.binseg.configs.models.driu',
          'HED = bob.ip.binseg.configs.models.hed',
          'M2UNet = bob.ip.binseg.configs.models.m2unet',
          'UNet = bob.ip.binseg.configs.models.unet',
          'ResUNet = bob.ip.binseg.configs.models.resunet',
          'DRIUADABOUND = bob.ip.binseg.configs.models.driuadabound',
          'DRIVETRAIN = bob.ip.binseg.configs.datasets.drivetrain',
          'DRIVECROPTRAIN = bob.ip.binseg.configs.datasets.drivecroptrain',
          'DRIVECROPTEST = bob.ip.binseg.configs.datasets.drivecroptest',
          'DRIVETEST = bob.ip.binseg.configs.datasets.drivetest',
          ]
    },



    # check classifiers, add and remove as you see fit
    # full list here: https://pypi.org/classifiers/
    # don't remove the Bob framework unless it's not a bob package
    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

)