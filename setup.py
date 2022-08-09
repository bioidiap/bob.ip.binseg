from setuptools import dist, setup

from bob.extension.utils import find_packages, load_requirements

dist.Distribution(dict(setup_requires=["bob.extension"]))
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
            "train_detection = bob.ip.binseg.script.detection_train:train",
            "predict = bob.ip.binseg.script.predict:predict",
            "evaluate = bob.ip.binseg.script.evaluate:evaluate",
            "compare =  bob.ip.binseg.script.compare:compare",
            "significance =  bob.ip.binseg.script.significance:significance",
            "analyze =  bob.ip.binseg.script.analyze:analyze",
            "experiment =  bob.ip.binseg.script.experiment:experiment",
            "mkmask = bob.ip.binseg.script.mkmask:mkmask",
            "train-analysis = bob.ip.binseg.script.train_analysis:train_analysis",
        ],
        # bob train configurations
        "bob.ip.binseg.config": [
            # models
            "driu = bob.ip.binseg.configs.models.driu",
            "driu-bn = bob.ip.binseg.configs.models.driu_bn",
            "driu-od = bob.ip.binseg.configs.models.driu_od",
            "hed = bob.ip.binseg.configs.models.hed",
            "m2unet = bob.ip.binseg.configs.models.m2unet",
            "unet = bob.ip.binseg.configs.models.unet",
            "resunet = bob.ip.binseg.configs.models.resunet",
            "lwnet = bob.ip.binseg.configs.models.lwnet",
            # example datasets
            "csv-dataset-example = bob.ip.binseg.configs.datasets.csv",
            # drive dataset - retinography
            "drive = bob.ip.binseg.configs.datasets.drive.default",
            "drive-768 = bob.ip.binseg.configs.datasets.drive.default_768",
            "drive-1024 = bob.ip.binseg.configs.datasets.drive.default_1024",
            "drive-2nd = bob.ip.binseg.configs.datasets.drive.second_annotator",
            "drive-xtest = bob.ip.binseg.configs.datasets.drive.xtest",
            "drive-mtest = bob.ip.binseg.configs.datasets.drive.mtest",
            "drive-covd = bob.ip.binseg.configs.datasets.drive.covd",
            # drhagis dataset - retinography
            "drhagis = bob.ip.binseg.configs.datasets.drhagis.default",
            # stare dataset - retinography
            "stare = bob.ip.binseg.configs.datasets.stare.ah",
            "stare-768 = bob.ip.binseg.configs.datasets.stare.ah_768",
            "stare-1024 = bob.ip.binseg.configs.datasets.stare.ah_1024",
            "stare-2nd = bob.ip.binseg.configs.datasets.stare.vk",
            "stare-xtest = bob.ip.binseg.configs.datasets.stare.xtest",
            "stare-mtest = bob.ip.binseg.configs.datasets.stare.mtest",
            "stare-covd = bob.ip.binseg.configs.datasets.stare.covd",
            # iostar - retinography
            "iostar-vessel = bob.ip.binseg.configs.datasets.iostar.vessel",
            "iostar-vessel-768 = bob.ip.binseg.configs.datasets.iostar.vessel_768",
            "iostar-vessel-xtest = bob.ip.binseg.configs.datasets.iostar.vessel_xtest",
            "iostar-vessel-mtest = bob.ip.binseg.configs.datasets.iostar.vessel_mtest",
            "iostar-disc = bob.ip.binseg.configs.datasets.iostar.optic_disc",
            "iostar-disc-512 = bob.ip.binseg.configs.datasets.iostar.optic_disc_512",
            "iostar-disc-768 = bob.ip.binseg.configs.datasets.iostar.optic_disc_768",
            "iostar-vessel-covd = bob.ip.binseg.configs.datasets.iostar.covd",
            # hrf - retinography
            "hrf = bob.ip.binseg.configs.datasets.hrf.default",
            "hrf-768 = bob.ip.binseg.configs.datasets.hrf.default_768",
            "hrf-1024 = bob.ip.binseg.configs.datasets.hrf.default_1024",
            "hrf-xtest = bob.ip.binseg.configs.datasets.hrf.xtest",
            "hrf-mtest = bob.ip.binseg.configs.datasets.hrf.mtest",
            "hrf-highres = bob.ip.binseg.configs.datasets.hrf.default_fullres",
            "hrf-covd = bob.ip.binseg.configs.datasets.hrf.covd",
            # chase-db1 - retinography
            "chasedb1 = bob.ip.binseg.configs.datasets.chasedb1.first_annotator",
            "chasedb1-768 = bob.ip.binseg.configs.datasets.chasedb1.first_annotator_768",
            "chasedb1-1024 = bob.ip.binseg.configs.datasets.chasedb1.first_annotator_1024",
            "chasedb1-2nd = bob.ip.binseg.configs.datasets.chasedb1.second_annotator",
            "chasedb1-xtest = bob.ip.binseg.configs.datasets.chasedb1.xtest",
            "chasedb1-mtest = bob.ip.binseg.configs.datasets.chasedb1.mtest",
            "chasedb1-covd = bob.ip.binseg.configs.datasets.chasedb1.covd",
            # drionsdb - retinography
            "drionsdb = bob.ip.binseg.configs.datasets.drionsdb.expert1",
            "drionsdb-512 = bob.ip.binseg.configs.datasets.drionsdb.expert1_512",
            "drionsdb-768 = bob.ip.binseg.configs.datasets.drionsdb.expert1_768",
            "drionsdb-2nd = bob.ip.binseg.configs.datasets.drionsdb.expert2",
            "drionsdb-2nd-512 = bob.ip.binseg.configs.datasets.drionsdb.expert2_512",
            # drishti-gs1 - retinography
            "drishtigs1-disc = bob.ip.binseg.configs.datasets.drishtigs1.disc_all",
            "drishtigs1-disc-512 = bob.ip.binseg.configs.datasets.drishtigs1.disc_all_512",
            "drishtigs1-disc-768 = bob.ip.binseg.configs.datasets.drishtigs1.disc_all_768",
            "drishtigs1-cup = bob.ip.binseg.configs.datasets.drishtigs1.cup_all",
            "drishtigs1-cup-512 = bob.ip.binseg.configs.datasets.drishtigs1.cup_all_512",
            "drishtigs1-cup-768 = bob.ip.binseg.configs.datasets.drishtigs1.cup_all_768",
            "drishtigs1-disc-any = bob.ip.binseg.configs.datasets.drishtigs1.disc_any",
            "drishtigs1-cup-any = bob.ip.binseg.configs.datasets.drishtigs1.cup_any",
            # refuge - retinography
            "refuge-cup = bob.ip.binseg.configs.datasets.refuge.cup",
            "refuge-cup-512 = bob.ip.binseg.configs.datasets.refuge.cup_512",
            "refuge-cup-768 = bob.ip.binseg.configs.datasets.refuge.cup_768",
            "refuge-disc = bob.ip.binseg.configs.datasets.refuge.disc",
            "refuge-disc-512 = bob.ip.binseg.configs.datasets.refuge.disc_512",
            "refuge-disc-768 = bob.ip.binseg.configs.datasets.refuge.disc_768",
            # rim one r3 - retinography
            "rimoner3-cup = bob.ip.binseg.configs.datasets.rimoner3.cup_exp1",
            "rimoner3-disc = bob.ip.binseg.configs.datasets.rimoner3.disc_exp1",
            "rimoner3-cup-512 = bob.ip.binseg.configs.datasets.rimoner3.cup_exp1_512",
            "rimoner3-cup-768 = bob.ip.binseg.configs.datasets.rimoner3.cup_exp1_768",
            "rimoner3-disc-512 = bob.ip.binseg.configs.datasets.rimoner3.disc_exp1_512",
            "rimoner3-disc-768 = bob.ip.binseg.configs.datasets.rimoner3.disc_exp1_768",
            "rimoner3-cup-2nd = bob.ip.binseg.configs.datasets.rimoner3.cup_exp2",
            "rimoner3-disc-2nd = bob.ip.binseg.configs.datasets.rimoner3.disc_exp2",
            # combined vessels - retinography
            "combined-vessels = bob.ip.binseg.configs.datasets.combined.vessel",
            # combined discs - retinography
            "combined-disc = bob.ip.binseg.configs.datasets.combined.od",
            # combined cups - retinography
            "combined-cup = bob.ip.binseg.configs.datasets.combined.oc",
            # montgomery county - cxr
            "montgomery = bob.ip.binseg.configs.datasets.montgomery.default",
            "montgomery-xtest = bob.ip.binseg.configs.datasets.montgomery.xtest",
            # shenzhen - cxr
            "shenzhen = bob.ip.binseg.configs.datasets.shenzhen.default",
            "shenzhen-small = bob.ip.binseg.configs.datasets.shenzhen.default_256",
            "shenzhen-xtest = bob.ip.binseg.configs.datasets.shenzhen.xtest",
            # jsrt - cxr
            "jsrt = bob.ip.binseg.configs.datasets.jsrt.default",
            "jsrt-xtest = bob.ip.binseg.configs.datasets.jsrt.xtest",
            # cxr8 - cxr
            "cxr8 = bob.ip.binseg.configs.datasets.cxr8.default",
            "cxr8-idiap = bob.ip.binseg.configs.datasets.cxr8.idiap",
            "cxr8-xtest = bob.ip.binseg.configs.datasets.cxr8.xtest",
            "cxr8-idiap-xtest = bob.ip.binseg.configs.datasets.cxr8.xtest_idiap",
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
