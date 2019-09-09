#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, dist

dist.Distribution(dict(setup_requires=["bob.extension"]))

from bob.extension.utils import load_requirements, find_packages

install_requires = load_requirements()


setup(
    name="bob.ip.binseg",
    version=open("version.txt").read().rstrip(),
    description="Binary Segmentation Benchmark Package for Bob",
    url="https://gitlab.idiap.ch/bob/bob.ip.binseg",
    license="GPLv3",
    # there may be multiple authors (separate entries by comma)
    author="Tim Laibacher",
    author_email="tim.laibacher@idiap.ch",
    # there may be a maintainer apart from the author - you decide
    maintainer="Andre Anjos",
    maintainer_email="andre.anjos@idiap.ch",
    # you may add more keywords separating those by commas (a, b, c, ...)
    keywords="bob",
    long_description=open("README.rst").read(),
    # leave this here, it is pretty standard
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        # main entry for bob binseg cli
        "bob.cli": ["binseg = bob.ip.binseg.script.binseg:binseg"],
        # bob hed sub-commands
        "bob.ip.binseg.cli": [
            "train = bob.ip.binseg.script.binseg:train",
            "test = bob.ip.binseg.script.binseg:test",
            "compare =  bob.bin.binseg.script.binseg:compare",
            "gridtable = bob.ip.binseg.script.binseg:testcheckpoints",
            "visualize = bob.ip.binseg.script.binseg:visualize",
        ],
        # bob train configurations
        "bob.ip.binseg.config": [
            "DRIU = bob.ip.binseg.configs.models.driu",
            "DRIUBN = bob.ip.binseg.configs.models.driubn",
            "DRIUSSL = bob.ip.binseg.configs.models.driussl",
            "DRIUBNSSL = bob.ip.binseg.configs.models.driubnssl",
            "DRIUOD = bob.ip.binseg.configs.models.driuod",
            "HED = bob.ip.binseg.configs.models.hed",
            "M2UNet = bob.ip.binseg.configs.models.m2unet",
            "M2UNetSSL = bob.ip.binseg.configs.models.m2unetssl",
            "UNet = bob.ip.binseg.configs.models.unet",
            "ResUNet = bob.ip.binseg.configs.models.resunet",
            "IMAGEFOLDER = bob.ip.binseg.configs.datasets.imagefolder",
            "CHASEDB1 = bob.ip.binseg.configs.datasets.chasedb1",
            "CHASEDB1TEST = bob.ip.binseg.configs.datasets.chasedb1test",
            "COVD-DRIVE = bob.ip.binseg.configs.datasets.starechasedb1iostarhrf544",
            "COVD-DRIVE_SSL = bob.ip.binseg.configs.datasets.starechasedb1iostarhrf544ssldrive",
            "COVD-STARE = bob.ip.binseg.configs.datasets.drivechasedb1iostarhrf608",
            "COVD-STARE_SSL = bob.ip.binseg.configs.datasets.drivechasedb1iostarhrf608sslstare",
            "COVD-IOSTARVESSEL = bob.ip.binseg.configs.datasets.drivestarechasedb1hrf1024",
            "COVD-IOSTARVESSEL_SSL = bob.ip.binseg.configs.datasets.drivestarechasedb1hrf1024ssliostar",
            "COVD-HRF = bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168",
            "COVD-HRF_SSL = bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168sslhrf",
            "COVD-CHASEDB1 = bob.ip.binseg.configs.datasets.drivestareiostarhrf960",
            "COVD-CHASEDB1_SSL = bob.ip.binseg.configs.datasets.drivestareiostarhrf960sslchase",
            "DRIONSDB = bob.ip.binseg.configs.datasets.drionsdb",
            "DRIONSDBTEST = bob.ip.binseg.configs.datasets.drionsdbtest",
            "DRISHTIGS1OD = bob.ip.binseg.configs.datasets.dristhigs1od",
            "DRISHTIGS1ODTEST = bob.ip.binseg.configs.datasets.dristhigs1odtest",
            "DRISHTIGS1CUP = bob.ip.binseg.configs.datasets.dristhigs1cup",
            "DRISHTIGS1CUPTEST = bob.ip.binseg.configs.datasets.dristhigs1cuptest",
            "DRIVE = bob.ip.binseg.configs.datasets.drive",
            "DRIVETEST = bob.ip.binseg.configs.datasets.drivetest",
            "HRF = bob.ip.binseg.configs.datasets.hrf1168",
            "HRFTEST = bob.ip.binseg.configs.datasets.hrftest",
            "IOSTAROD = bob.ip.binseg.configs.datasets.iostarod",
            "IOSTARODTEST = bob.ip.binseg.configs.datasets.iostarodtest",
            "IOSTARVESSEL = bob.ip.binseg.configs.datasets.iostarvessel",
            "IOSTARVESSELTEST = bob.ip.binseg.configs.datasets.iostarvesseltest",
            "REFUGECUP = bob.ip.binseg.configs.datasets.refugecup",
            "REFUGECUPTEST = bob.ip.binseg.configs.datasets.refugecuptest",
            "REFUGEOD = bob.ip.binseg.configs.datasets.refugeod",
            "REFUGEODTEST = bob.ip.binseg.configs.datasets.refugeodtest",
            "RIMONER3CUP = bob.ip.binseg.configs.datasets.rimoner3cup",
            "RIMONER3CUPTEST = bob.ip.binseg.configs.datasets.rimoner3cuptest",
            "RIMONER3OD = bob.ip.binseg.configs.datasets.rimoner3od",
            "RIMONER3ODTEST = bob.ip.binseg.configs.datasets.rimoner3odtest",
            "STARE = bob.ip.binseg.configs.datasets.stare",
            "STARETEST = bob.ip.binseg.configs.datasets.staretest",
        ],
    },
    # check classifiers, add and remove as you see fit
    # full list here: https://pypi.org/classifiers/
    # don't remove the Bob framework unless it's not a bob package
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
