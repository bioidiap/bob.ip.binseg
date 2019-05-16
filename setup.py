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
          'pdfoverview = bob.ip.binseg.script.binseg:testcheckpoints',
          'gridtable = bob.ip.binseg.script.binseg:testcheckpoints',
          'visualize = bob.ip.binseg.script.binseg:visualize',
        ],

         #bob hed train configurations
        'bob.ip.binseg.config': [
          'DRIU = bob.ip.binseg.configs.models.driu',
          'DRIUJ = bob.ip.binseg.configs.models.driuj',
          'DRIUJ7 = bob.ip.binseg.configs.models.driuj7',
          'DRIUODJ = bob.ip.binseg.configs.models.driuodj',
          'HED = bob.ip.binseg.configs.models.hed',
          'HEDJ = bob.ip.binseg.configs.models.hedj',
          'HEDJ7 = bob.ip.binseg.configs.models.hedj7',
          'HEDODJ = bob.ip.binseg.configs.models.hedodj',
          'M2UNet = bob.ip.binseg.configs.models.m2unet',
          'M2UNetJ = bob.ip.binseg.configs.models.m2unetj',
          'M2UNetJ7 = bob.ip.binseg.configs.models.m2unetj7',
          'UNet = bob.ip.binseg.configs.models.unet',
          'UNetJ = bob.ip.binseg.configs.models.unetj',
          'UNetJ7 = bob.ip.binseg.configs.models.unetj7',
          'UNetODJ = bob.ip.binseg.configs.models.unetodj',
          'ResUNet = bob.ip.binseg.configs.models.resunet',
          'ResUNetJ = bob.ip.binseg.configs.models.resunetj',
          'ShapeResUNet = bob.ip.binseg.configs.models.shaperesunet',
          'CHASEDB1 = bob.ip.binseg.configs.datasets.chasedb1',
          'CHASEDB1TEST = bob.ip.binseg.configs.datasets.chasedb1test',
          'DRIONSDB = bob.ip.binseg.configs.datasets.drionsdb',
          'DRIONSDBTEST = bob.ip.binseg.configs.datasets.drionsdbtest',
          'DRISHTIGS1OD = bob.ip.binseg.configs.datasets.dristhigs1od',
          'DRISHTIGS1ODTEST = bob.ip.binseg.configs.datasets.dristhigs1odtest',
          'DRISHTIGS1CUP = bob.ip.binseg.configs.datasets.dristhigs1cup',
          'DRISHTIGS1CUPTEST = bob.ip.binseg.configs.datasets.dristhigs1cuptest',
          'DRIVE = bob.ip.binseg.configs.datasets.drive',
          'DRIVETEST = bob.ip.binseg.configs.datasets.drivetest',
          'HRF = bob.ip.binseg.configs.datasets.hrf',
          'HRFTEST = bob.ip.binseg.configs.datasets.hrftest',
          'IOSTAROD = bob.ip.binseg.configs.datasets.iostarod',
          'IOSTARODTEST = bob.ip.binseg.configs.datasets.iostarodtest',
          'IOSTARVESSEL = bob.ip.binseg.configs.datasets.iostarvessel',
          'IOSTARVESSELTEST = bob.ip.binseg.configs.datasets.iostarvesseltest',
          'REFUGECUP = bob.ip.binseg.configs.datasets.refugecup',
          'REFUGECUPTEST = bob.ip.binseg.configs.datasets.refugecuptest',
          'REFUGEOD = bob.ip.binseg.configs.datasets.refugeod',
          'REFUGEODTEST = bob.ip.binseg.configs.datasets.refugeodtest',
          'RIMONER3CUP = bob.ip.binseg.configs.datasets.rimoner3cup',
          'RIMONER3CUPTEST = bob.ip.binseg.configs.datasets.rimoner3cuptest',
          'RIMONER3OD = bob.ip.binseg.configs.datasets.rimoner3od',
          'RIMONER3ODTEST = bob.ip.binseg.configs.datasets.rimoner3odtest',
          'STARE = bob.ip.binseg.configs.datasets.stare',
          'STARETEST = bob.ip.binseg.configs.datasets.staretest',
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