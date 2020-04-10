.. -*- coding: utf-8 -*-

=====================================
 Application Program Interface (API)
=====================================

.. To update these lists, run the following command on the root of the package:
.. find bob -name '*.py' | sed -e 's#/#.#g;s#.py$##g;s#.__init__##g' | sort
.. You may apply further filtering to update only one of the subsections below


Data Manipulation
-----------------

.. autosummary::
   :toctree: api/data

   bob.ip.binseg.data.binsegdataset
   bob.ip.binseg.data.folderdataset
   bob.ip.binseg.data.csvdataset
   bob.ip.binseg.data.jsondataset
   bob.ip.binseg.data.loader
   bob.ip.binseg.data.sample
   bob.ip.binseg.data.utils
   bob.ip.binseg.data.transforms


Datasets
--------

.. autosummary::
   :toctree: api/dataset

   bob.ip.binseg.data.drive
   bob.ip.binseg.data.stare
   bob.ip.binseg.data.chasedb1


Engines
-------

.. autosummary::
   :toctree: api/engine

   bob.ip.binseg.engine
   bob.ip.binseg.engine.trainer
   bob.ip.binseg.engine.ssltrainer
   bob.ip.binseg.engine.predictor
   bob.ip.binseg.engine.evaluator
   bob.ip.binseg.engine.adabound


Neural Network Models
---------------------

.. autosummary::
   :toctree: api/modeling

   bob.ip.binseg.modeling
   bob.ip.binseg.modeling.backbones
   bob.ip.binseg.modeling.backbones.mobilenetv2
   bob.ip.binseg.modeling.backbones.resnet
   bob.ip.binseg.modeling.backbones.vgg
   bob.ip.binseg.modeling.driu
   bob.ip.binseg.modeling.driubn
   bob.ip.binseg.modeling.driuod
   bob.ip.binseg.modeling.driupix
   bob.ip.binseg.modeling.hed
   bob.ip.binseg.modeling.losses
   bob.ip.binseg.modeling.m2u
   bob.ip.binseg.modeling.make_layers
   bob.ip.binseg.modeling.resunet
   bob.ip.binseg.modeling.unet


Toolbox
-------

.. autosummary::
   :toctree: api/utils

   bob.ip.binseg.utils
   bob.ip.binseg.utils.checkpointer
   bob.ip.binseg.utils.metric
   bob.ip.binseg.utils.model_serialization
   bob.ip.binseg.utils.model_zoo
   bob.ip.binseg.utils.plot
   bob.ip.binseg.utils.summary


.. _bob.ip.binseg.configs:

Preset Configurations
---------------------

Preset configurations for baseline systems

This module contains preset configurations for baseline FCN architectures and
datasets.


Models
======

.. autosummary::
   :toctree: api/configs/models
   :template: config.rst

   bob.ip.binseg.configs.models.driu
   bob.ip.binseg.configs.models.driubn
   bob.ip.binseg.configs.models.driubnssl
   bob.ip.binseg.configs.models.driuod
   bob.ip.binseg.configs.models.driussl
   bob.ip.binseg.configs.models.hed
   bob.ip.binseg.configs.models.m2unet
   bob.ip.binseg.configs.models.m2unetssl
   bob.ip.binseg.configs.models.resunet
   bob.ip.binseg.configs.models.unet


.. _bob.ip.binseg.configs.datasets:

Datasets
========

.. autosummary::
   :toctree: api/configs/datasets
   :template: config.rst

   bob.ip.binseg.configs.datasets.chasedb1
   bob.ip.binseg.configs.datasets.chasedb11024
   bob.ip.binseg.configs.datasets.chasedb11168
   bob.ip.binseg.configs.datasets.chasedb1544
   bob.ip.binseg.configs.datasets.chasedb1608
   bob.ip.binseg.configs.datasets.chasedb1test
   bob.ip.binseg.configs.datasets.csv
   bob.ip.binseg.configs.datasets.drionsdb
   bob.ip.binseg.configs.datasets.drionsdbtest
   bob.ip.binseg.configs.datasets.dristhigs1cup
   bob.ip.binseg.configs.datasets.dristhigs1cuptest
   bob.ip.binseg.configs.datasets.dristhigs1od
   bob.ip.binseg.configs.datasets.dristhigs1odtest
   bob.ip.binseg.configs.datasets.drive
   bob.ip.binseg.configs.datasets.drive1024
   bob.ip.binseg.configs.datasets.drive1024test
   bob.ip.binseg.configs.datasets.drive1168
   bob.ip.binseg.configs.datasets.drive608
   bob.ip.binseg.configs.datasets.drive960
   bob.ip.binseg.configs.datasets.drivechasedb1iostarhrf608
   bob.ip.binseg.configs.datasets.drivechasedb1iostarhrf608sslstare
   bob.ip.binseg.configs.datasets.drivestarechasedb1hrf1024
   bob.ip.binseg.configs.datasets.drivestarechasedb1hrf1024ssliostar
   bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168
   bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168sslhrf
   bob.ip.binseg.configs.datasets.drivestareiostarhrf960
   bob.ip.binseg.configs.datasets.drivestareiostarhrf960sslchase
   bob.ip.binseg.configs.datasets.drivetest
   bob.ip.binseg.configs.datasets.folder
   bob.ip.binseg.configs.datasets.hrf
   bob.ip.binseg.configs.datasets.hrf1024
   bob.ip.binseg.configs.datasets.hrf1168
   bob.ip.binseg.configs.datasets.hrf1168test
   bob.ip.binseg.configs.datasets.hrf544
   bob.ip.binseg.configs.datasets.hrf544test
   bob.ip.binseg.configs.datasets.hrf608
   bob.ip.binseg.configs.datasets.hrf960
   bob.ip.binseg.configs.datasets.hrftest
   bob.ip.binseg.configs.datasets.iostarod
   bob.ip.binseg.configs.datasets.iostarodtest
   bob.ip.binseg.configs.datasets.iostarvessel
   bob.ip.binseg.configs.datasets.iostarvessel1168
   bob.ip.binseg.configs.datasets.iostarvessel544
   bob.ip.binseg.configs.datasets.iostarvessel544test
   bob.ip.binseg.configs.datasets.iostarvessel608
   bob.ip.binseg.configs.datasets.iostarvessel960
   bob.ip.binseg.configs.datasets.iostarvesseltest
   bob.ip.binseg.configs.datasets.refugecup
   bob.ip.binseg.configs.datasets.refugecuptest
   bob.ip.binseg.configs.datasets.refugeod
   bob.ip.binseg.configs.datasets.refugeodtest
   bob.ip.binseg.configs.datasets.rimoner3cup
   bob.ip.binseg.configs.datasets.rimoner3cuptest
   bob.ip.binseg.configs.datasets.rimoner3od
   bob.ip.binseg.configs.datasets.rimoner3odtest
   bob.ip.binseg.configs.datasets.stare
   bob.ip.binseg.configs.datasets.stare1024
   bob.ip.binseg.configs.datasets.stare1168
   bob.ip.binseg.configs.datasets.stare544
   bob.ip.binseg.configs.datasets.stare960
   bob.ip.binseg.configs.datasets.starechasedb1iostarhrf544
   bob.ip.binseg.configs.datasets.starechasedb1iostarhrf544ssldrive
   bob.ip.binseg.configs.datasets.staretest
