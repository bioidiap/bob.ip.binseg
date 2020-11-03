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
            "significance =  bob.ip.binseg.script.significance:significance",
            "analyze =  bob.ip.binseg.script.analyze:analyze",
            "experiment =  bob.ip.binseg.script.experiment:experiment",
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
            "lwnet = bob.ip.binseg.configs.models.lwnet",

            # example datasets
            "csv-dataset-example = bob.ip.binseg.configs.datasets.csv",

            # drive dataset
            "drive = bob.ip.binseg.configs.datasets.drive.default",
            "drive-2nd = bob.ip.binseg.configs.datasets.drive.second_annotator",
            "drive-xtest = bob.ip.binseg.configs.datasets.drive.xtest",
            "drive-mtest = bob.ip.binseg.configs.datasets.drive.mtest",
            "drive-covd = bob.ip.binseg.configs.datasets.drive.covd",
            "drive-ssl = bob.ip.binseg.configs.datasets.drive.ssl",

            # stare dataset
            "stare = bob.ip.binseg.configs.datasets.stare.ah",
            "stare-2nd = bob.ip.binseg.configs.datasets.stare.vk",
            "stare-xtest = bob.ip.binseg.configs.datasets.stare.xtest",
            "stare-mtest = bob.ip.binseg.configs.datasets.stare.mtest",
            "stare-covd = bob.ip.binseg.configs.datasets.stare.covd",
            "stare-ssl = bob.ip.binseg.configs.datasets.stare.ssl",

            # iostar
            "iostar-vessel = bob.ip.binseg.configs.datasets.iostar.vessel",
            "iostar-vessel-xtest = bob.ip.binseg.configs.datasets.iostar.vessel_xtest",
            "iostar-vessel-mtest = bob.ip.binseg.configs.datasets.iostar.vessel_mtest",
            "iostar-disc = bob.ip.binseg.configs.datasets.iostar.optic_disc",
            "iostar-vessel-covd = bob.ip.binseg.configs.datasets.iostar.covd",
            "iostar-vessel-ssl = bob.ip.binseg.configs.datasets.iostar.ssl",

            # hrf
            "hrf = bob.ip.binseg.configs.datasets.hrf.default",
            "hrf-xtest = bob.ip.binseg.configs.datasets.hrf.xtest",
            "hrf-mtest = bob.ip.binseg.configs.datasets.hrf.mtest",
            "hrf-highres = bob.ip.binseg.configs.datasets.hrf.default_fullres",
            "hrf-covd = bob.ip.binseg.configs.datasets.hrf.covd",
            "hrf-ssl = bob.ip.binseg.configs.datasets.hrf.ssl",

            # chase-db1
            "chasedb1 = bob.ip.binseg.configs.datasets.chasedb1.first_annotator",
            "chasedb1-2nd = bob.ip.binseg.configs.datasets.chasedb1.second_annotator",
            "chasedb1-xtest = bob.ip.binseg.configs.datasets.chasedb1.xtest",
            "chasedb1-mtest = bob.ip.binseg.configs.datasets.chasedb1.mtest",
            "chasedb1-covd = bob.ip.binseg.configs.datasets.chasedb1.covd",
            "chasedb1-ssl = bob.ip.binseg.configs.datasets.chasedb1.ssl",

            # drionsdb
            "drionsdb = bob.ip.binseg.configs.datasets.drionsdb.expert1",
            "drionsdb-2nd = bob.ip.binseg.configs.datasets.drionsdb.expert2",

            # drishti-gs1
            "drishtigs1-disc = bob.ip.binseg.configs.datasets.drishtigs1.disc_all",
            "drishtigs1-cup = bob.ip.binseg.configs.datasets.drishtigs1.cup_all",
            "drishtigs1-disc-any = bob.ip.binseg.configs.datasets.drishtigs1.disc_any",
            "drishtigs1-cup-any = bob.ip.binseg.configs.datasets.drishtigs1.cup_any",

            # refuge
            "refuge-cup = bob.ip.binseg.configs.datasets.refuge.cup",
            "refuge-disc = bob.ip.binseg.configs.datasets.refuge.disc",

            # rim one r3
            "rimoner3-cup = bob.ip.binseg.configs.datasets.rimoner3.cup_exp1",
            "rimoner3-disc = bob.ip.binseg.configs.datasets.rimoner3.disc_exp1",
            "rimoner3-cup-2nd = bob.ip.binseg.configs.datasets.rimoner3.cup_exp2",
            "rimoner3-disc-2nd = bob.ip.binseg.configs.datasets.rimoner3.disc_exp2",
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
