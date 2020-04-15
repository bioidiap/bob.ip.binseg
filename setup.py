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
        # bob binseg sub-commands
        "bob.ip.binseg.cli": [
            "config = bob.ip.binseg.script.config:config",
            "dataset =  bob.ip.binseg.script.dataset:dataset",
            "train = bob.ip.binseg.script.train:train",
            "predict = bob.ip.binseg.script.predict:predict",
            "evaluate = bob.ip.binseg.script.evaluate:evaluate",
            "compare =  bob.ip.binseg.script.compare:compare",
        ],
        # bob train configurations
        "bob.ip.binseg.config": [

            # models
            "driu = bob.ip.binseg.configs.models.driu",
            "driu-bn = bob.ip.binseg.configs.models.driu_bn",
            "driu-ssl = bob.ip.binseg.configs.models.driu_ssl",
            "driu-bn-ssl = bob.ip.binseg.configs.models.driu_bn_ssl",
            "driu-od = bob.ip.binseg.configs.models.driu_od",
            "hed = bob.ip.binseg.configs.models.hed",
            "m2unet = bob.ip.binseg.configs.models.m2unet",
            "m2unet-ssl = bob.ip.binseg.configs.models.m2unet_ssl",
            "unet = bob.ip.binseg.configs.models.unet",
            "resunet = bob.ip.binseg.configs.models.resunet",

            # example datasets
            "csv-dataset-example = bob.ip.binseg.configs.datasets.csv",

            # drive dataset
            "drive = bob.ip.binseg.configs.datasets.drive",
            "covd-drive = bob.ip.binseg.configs.datasets.covd_drive",
            "covd-drive-ssl = bob.ip.binseg.configs.datasets.covd_drive_ssl",
            "drive-test = bob.ip.binseg.configs.datasets.drive_test",

            # stare dataset
            "stare = bob.ip.binseg.configs.datasets.stare",
            "covd-stare = bob.ip.binseg.configs.datasets.covd_stare",
            "covd-stare-ssl = bob.ip.binseg.configs.datasets.covd_stare_ssl",
            "stare-test = bob.ip.binseg.configs.datasets.stare_test",

            # iostar vessel
            "iostar-vessel = bob.ip.binseg.configs.datasets.iostar_vessel",
            "covd-iostar-vessel = bob.ip.binseg.configs.datasets.covd_iostar_vessel",
            "covd-iostar-vessel-ssl = bob.ip.binseg.configs.datasets.covd_iostar_vessel_ssl",
            "iostar-vessel-test = bob.ip.binseg.configs.datasets.iostar_vessel_test",

            # iostar optic disc
            "iostar-optic-disc = bob.ip.binseg.configs.datasets.iostar_od",
            "iostar-optic-disc-test = bob.ip.binseg.configs.datasets.iostar_od_test",

            # hrf (numbers represent target resolution)
            "hrf = bob.ip.binseg.configs.datasets.hrf_1168",
            "covd-hrf = bob.ip.binseg.configs.datasets.covd_hrf",
            "covd-hrf-ssl = bob.ip.binseg.configs.datasets.covd_hrf_ssl",
            "hrftest-test = bob.ip.binseg.configs.datasets.hrf_1168_test",

            # chase-db1
            "chasedb1 = bob.ip.binseg.configs.datasets.chasedb1",
            "covd-chasedb1 = bob.ip.binseg.configs.datasets.covd_chasedb1",
            "covd-chasedb1-ssl = bob.ip.binseg.configs.datasets.covd_chasedb1_ssl",
            "chasedb1-test = bob.ip.binseg.configs.datasets.chasedb1_test",

            # drionsdb
            "drionsdb = bob.ip.binseg.configs.datasets.drionsdb",
            "drionsdb-test = bob.ip.binseg.configs.datasets.drionsdb_test",

            # drishtigs
            "drishtigs1-od = bob.ip.binseg.configs.datasets.dristhigs1_od",
            "drishtigs1-od-test = bob.ip.binseg.configs.datasets.dristhigs1_od_test",
            "drishtigs1-cup = bob.ip.binseg.configs.datasets.dristhigs1_cup",
            "drishtigs1-cup-test = bob.ip.binseg.configs.datasets.dristhigs1_cup_test",
            # refuge
            "refuge-cup = bob.ip.binseg.configs.datasets.refuge_cup",
            "refuge-cup-dev = bob.ip.binseg.configs.datasets.refuge_cup_dev",
            "refuge-cup-test = bob.ip.binseg.configs.datasets.refuge_cup_test",
            "refuge-od = bob.ip.binseg.configs.datasets.refuge_od",
            "refuge-od-dev = bob.ip.binseg.configs.datasets.refuge_od_dev",
            "refuge-od-test = bob.ip.binseg.configs.datasets.refuge_od_test",

            # rim one r3
            "rimoner3-cup = bob.ip.binseg.configs.datasets.rimoner3_cup",
            "rimoner3-cup-test = bob.ip.binseg.configs.datasets.rimoner3_cup_test",
            "rimoner3-od = bob.ip.binseg.configs.datasets.rimoner3_od",
            "rimoner3-od-test = bob.ip.binseg.configs.datasets.rimoner3_od_test",
        ],
    },
    # check classifiers, add and remove as you see fit
    # full list here: https://pypi.org/classifiers/
    # don't remove the Bob framework unless it's not a bob package
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
