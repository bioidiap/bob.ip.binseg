.. -*- coding: utf-8 -*-

=====
 API
=====

.. To update these lists, run the following command on the root of the package:
.. find bob -name '*.py' | sed -e 's#/#.#g;s#.py$##g;s#.__init__##g' | sort
.. You may apply further filtering to update only one of the subsections below

.. autosummary::
   :toctree: api/base

   bob.ip.binseg


Data Manipulation
-----------------

.. autosummary::
   :toctree: api/data

   bob.ip.binseg.data
   bob.ip.binseg.data.binsegdataset
   bob.ip.binseg.data.imagefolder
   bob.ip.binseg.data.imagefolderinference
   bob.ip.binseg.data.transforms


Engines
-------

.. autosummary::
   :toctree: api/engine

   bob.ip.binseg.engine
   bob.ip.binseg.engine.adabound
   bob.ip.binseg.engine.inferencer
   bob.ip.binseg.engine.predicter
   bob.ip.binseg.engine.ssltrainer
   bob.ip.binseg.engine.trainer


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
   bob.ip.binseg.utils.click
   bob.ip.binseg.utils.evaluate
   bob.ip.binseg.utils.metric
   bob.ip.binseg.utils.model_serialization
   bob.ip.binseg.utils.model_zoo
   bob.ip.binseg.utils.plot
   bob.ip.binseg.utils.rsttable
   bob.ip.binseg.utils.summary
   bob.ip.binseg.utils.transformfolder


Scripts
-------

.. autosummary::
   :toctree: api/scripts

   bob.ip.binseg.script
   bob.ip.binseg.script.binseg


Preset Configurations
---------------------


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


Datasets
========

.. autosummary::
   :toctree: api/configs/datasets
   :template: config.rst

   bob.ip.binseg.configs.datasets.amdrive
   bob.ip.binseg.configs.datasets.amdrivetest
   bob.ip.binseg.configs.datasets.chasedb1
   bob.ip.binseg.configs.datasets.chasedb11024
   bob.ip.binseg.configs.datasets.chasedb11168
   bob.ip.binseg.configs.datasets.chasedb1544
   bob.ip.binseg.configs.datasets.chasedb1608
   bob.ip.binseg.configs.datasets.chasedb1test
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
   bob.ip.binseg.configs.datasets.drivestarechasedb11168
   bob.ip.binseg.configs.datasets.drivestarechasedb1hrf1024
   bob.ip.binseg.configs.datasets.drivestarechasedb1hrf1024ssliostar
   bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168
   bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168sslhrf
   bob.ip.binseg.configs.datasets.drivestareiostarhrf960
   bob.ip.binseg.configs.datasets.drivestareiostarhrf960sslchase
   bob.ip.binseg.configs.datasets.drivetest
   bob.ip.binseg.configs.datasets.hrf
   bob.ip.binseg.configs.datasets.hrf1024
   bob.ip.binseg.configs.datasets.hrf1168
   bob.ip.binseg.configs.datasets.hrf1168test
   bob.ip.binseg.configs.datasets.hrf544
   bob.ip.binseg.configs.datasets.hrf544test
   bob.ip.binseg.configs.datasets.hrf608
   bob.ip.binseg.configs.datasets.hrf960
   bob.ip.binseg.configs.datasets.hrftest
   bob.ip.binseg.configs.datasets.imagefolder
   bob.ip.binseg.configs.datasets.imagefolderinference
   bob.ip.binseg.configs.datasets.imagefoldertest
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


Test Units
----------

.. autosummary::
   :toctree: api/tests

   bob.ip.binseg.test
   bob.ip.binseg.test.test_basemetrics
   bob.ip.binseg.test.test_batchmetrics
   bob.ip.binseg.test.test_checkpointer
   bob.ip.binseg.test.test_summary
   bob.ip.binseg.test.test_transforms
