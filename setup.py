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
            "compare =  bob.bin.binseg.script.binseg:compare",
            "evalpred = bob.ip.binseg.script.binseg:evalpred",
            "gridtable = bob.ip.binseg.script.binseg:testcheckpoints",
            "visualize = bob.ip.binseg.script.binseg:visualize",
            "config = bob.ip.binseg.script.config:config",
            "train = bob.ip.binseg.script.train:train",
            "predict = bob.ip.binseg.script.predict:predict",
            "evaluate = bob.ip.binseg.script.evaluate:evaluate",
        ],
        # bob train configurations
        "bob.ip.binseg.config": [

            # models
            "driu = bob.ip.binseg.configs.models.driu",
            "driu-bn = bob.ip.binseg.configs.models.driubn",
            "driu-ssl = bob.ip.binseg.configs.models.driussl",
            "driu-bn-ssl = bob.ip.binseg.configs.models.driubnssl",
            "driu-od = bob.ip.binseg.configs.models.driuod",
            "hed = bob.ip.binseg.configs.models.hed",
            "m2unet = bob.ip.binseg.configs.models.m2unet",
            "m2unet-ssl = bob.ip.binseg.configs.models.m2unetssl",
            "unet = bob.ip.binseg.configs.models.unet",
            "resunet = bob.ip.binseg.configs.models.resunet",

            # datasets
            "csv-dataset-example = bob.ip.binseg.configs.datasets.csv",
            "folder-dataset-example = bob.ip.binseg.configs.datasets.folder",

            # drive dataset (numbers represent target resolution)
            "drive = bob.ip.binseg.configs.datasets.drive",
            "covd-drive = bob.ip.binseg.configs.datasets.starechasedb1iostarhrf544",
            "covd-drive-ssl = bob.ip.binseg.configs.datasets.starechasedb1iostarhrf544ssldrive",
            "drive-test = bob.ip.binseg.configs.datasets.drivetest",

            # stare dataset (numbers represent target resolution)
            "stare = bob.ip.binseg.configs.datasets.stare",
            "covd-stare = bob.ip.binseg.configs.datasets.drivechasedb1iostarhrf608",
            "covd-stare-ssl = bob.ip.binseg.configs.datasets.drivechasedb1iostarhrf608sslstare",
            "stare-test = bob.ip.binseg.configs.datasets.staretest",

            # iostar vessel (numbers represent target resolution)
            "iostar-vessel = bob.ip.binseg.configs.datasets.iostarvessel",
            "covd-iostar-vessel = bob.ip.binseg.configs.datasets.drivestarechasedb1hrf1024",
            "covd-iostar-vessel-ssl = bob.ip.binseg.configs.datasets.drivestarechasedb1hrf1024ssliostar",
            "iostar-vessel-test = bob.ip.binseg.configs.datasets.iostarvesseltest",

            # iostar optic disc
            "iostarod = bob.ip.binseg.configs.datasets.iostarod",
            "iostarodtest = bob.ip.binseg.configs.datasets.iostarodtest",

            # hrf (numbers represent target resolution)
            "hrf = bob.ip.binseg.configs.datasets.hrf1168",
            "covd-hrf = bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168",
            "covd-hrf-ssl = bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168sslhrf",
            "hrftest-test = bob.ip.binseg.configs.datasets.hrftest",

            # chase-db1 (numbers represent target resolution)
            "chasedb1 = bob.ip.binseg.configs.datasets.chasedb1",
            "covd-chasedb1 = bob.ip.binseg.configs.datasets.drivestareiostarhrf960",
            "covd-chasedb1-ssl = bob.ip.binseg.configs.datasets.drivestareiostarhrf960sslchase",
            "chasedb1-test = bob.ip.binseg.configs.datasets.chasedb1test",

            # drionsdb
            "drionsdb = bob.ip.binseg.configs.datasets.drionsdb",
            "drionsdb-test = bob.ip.binseg.configs.datasets.drionsdbtest",

            # drishtigs
            "drishtigs1-od = bob.ip.binseg.configs.datasets.dristhigs1od",
            "drishtigs1-od-test = bob.ip.binseg.configs.datasets.dristhigs1odtest",
            "drishtigs1-cup = bob.ip.binseg.configs.datasets.dristhigs1cup",
            "drishtigs1-cup-test = bob.ip.binseg.configs.datasets.dristhigs1cuptest",
            # refuge
            "refuge-cup = bob.ip.binseg.configs.datasets.refugecup",
            "refuge-cup-test = bob.ip.binseg.configs.datasets.refugecuptest",
            "refuge-od = bob.ip.binseg.configs.datasets.refugeod",
            "refuge-od-test = bob.ip.binseg.configs.datasets.refugeodtest",

            # rim one r3
            "rimoner3-cup = bob.ip.binseg.configs.datasets.rimoner3cup",
            "rimoner3-cup-test = bob.ip.binseg.configs.datasets.rimoner3cuptest",
            "rimoner3-od = bob.ip.binseg.configs.datasets.rimoner3od",
            "rimoner3-od-test = bob.ip.binseg.configs.datasets.rimoner3odtest",
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
