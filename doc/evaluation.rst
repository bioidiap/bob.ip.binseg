.. -*- coding: utf-8 -*-
.. _bob.ip.binseg.evaluation:

==========
Evaluation
==========

To evaluate trained models use use ``bob binseg test`` followed by
the model config, the dataset config and the path to the pretrained
model via the argument ``-w``.

Alternatively point to the output folder used during training via
the ``-o`` argument. The Checkpointer will load the model as indicated
in the file: ``last_checkpoint``.

Use ``bob binseg test --help`` for more information.

E.g. run inference on model M2U-Net on the DRIVE test set:

.. code-block:: bash

    # Point directly to saved model via -w argument:
    bob binseg test M2UNet DRIVETEST -o /outputfolder/for/results -w /direct/path/to/weight/model_final.pth

    # Use training output path (requries last_checkpoint file to be present)
    # The evaluation results will be stored in the same folder
    bob binseg test M2UNet DRIVETEST -o /DRIVE/M2UNet/output


Pretrained Models
=================

Due to storage limitations we only provide weights of a subset
of all evaluated models, namely all DRIU and M2U-Net variants:


https://dl.dropboxusercontent.com/s/wnyjzmhhep2smjl/retinanet_MobileNetV2-FPN_1x.pth

+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
|                    | DRIU                                                                                                 | M2UNet                                                                                                             |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| DRIVE              | `DRIU_DRIVE.pth <https://dl.dropboxusercontent.com/s/rggn9ebj38c06uf/DRIU_DRIVE.pth>`_               | `M2UNet_DRIVE.pth <https://dl.dropboxusercontent.com/s/55xply8jm0g2skp/M2UNet_DRIVE.pth>`_                         |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| COVD-DRIVE         |                                                                                                      |                                                                                                                    |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| COVD-DRIVE SSL     |                                                                                                      |                                                                                                                    |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| STARE              | `DRIU_STARE.pth <https://dl.dropboxusercontent.com/s/sw5ivfzgz5djirc/DRIU_STARE.pth>`_               | `M2UNet_STARE.pth <https://dl.dropboxusercontent.com/s/pc9wb8r7tjvg06p/M2UNet_STARE.pth>`_                         |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| COVD-STARE         |                                                                                                      | `M2UNet_COVD-STARE.pth <https://dl.dropboxusercontent.com/s/vh1trws2nxqt65y/M2UNet_COVD-STARE.pth>`_               |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| COVD-STARE SSL     |                                                                                                      | `M2UNet_COVD-STARE_SSL.pth <https://dl.dropboxusercontent.com/s/slcvfgf1saf7t19/M2UNet_COVD-STARE_SSL.pth>`_       |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| CHASE_DB1          | `DRIU_CHASEDB1.pth <https://dl.dropboxusercontent.com/s/15gxvhdtq0gw074/DRIU_CHASEDB1.pth>`_         | `M2UNet_CHASEDB1.pth <https://dl.dropboxusercontent.com/s/jqq0z9boi17nhqf/M2UNet_CHASEDB1.pth>`_                   |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| COVD-CHASE_DB1     |                                                                                                      | `M2UNet_COVD-CHASEDB1.pth <https://dl.dropboxusercontent.com/s/pvbp0qky13q5o11/M2UNet_COVD-CHASEDB1.pth>`_         |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| COVD-CHASE_DB1 SSL |                                                                                                      | `M2UNet_COVD-CHASEDB1_SSL.pth <https://dl.dropboxusercontent.com/s/qx7mm5h8ywm98fi/M2UNet_COVD-CHASEDB1_SSL.pth>`_ |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| IOSTARVESSEL       | `DRIU_IOSTARVESSEL.pth <https://dl.dropboxusercontent.com/s/dx1dp8g4nct5r2z/DRIU_IOSTARVESSEL.pth>`_ | `M2UNet_IOSTARVESSEL.pth <https://dl.dropboxusercontent.com/s/g9jyvar9x8vvihr/M2UNet_IOSTARVESSEL.pth>`_           |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| COVD-IOSTAR        |                                                                                                      | `M2UNet_COVD-IOSTAR.pth <https://dl.dropboxusercontent.com/s/t5b2qomq6ey8i9t/M2UNet_COVD-IOSTAR.pth>`_             |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| COVD-IOSTAR SSL    |                                                                                                      | `M2UNet_COVD-IOSTAR_SSL.pth <https://dl.dropboxusercontent.com/s/70ynm2k3bpkj4mq/M2UNet_COVD-IOSTAR_SSL.pth>`_     |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| HRF                | `DRIU_HRF1168.pth <https://dl.dropboxusercontent.com/s/c02m2zyby1zndqx/DRIU_HRF1168.pth>`_           | `M2UNet_HRF1168.pth <https://dl.dropboxusercontent.com/s/g34g6nai1rsgbsc/M2UNet_HRF1168.pth>`_                     |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| COVD-HRF           |                                                                                                      | `M2UNet_COVD-HRF.pth <https://dl.dropboxusercontent.com/s/o3edhljeidl6fvi/M2UNet_COVD-HRF.pth>`_                   |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| COVD-HRF SSL       |                                                                                                      | `M2UNet_COVD-HRF_SSL.pth <https://dl.dropboxusercontent.com/s/2e0aq8a5vbop2yx/M2UNet_COVD-HRF_SSL.pth>`_           |
+--------------------+------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+



To run evaluation of pretrained models pass url as ``-w`` argument. E.g.:

.. code-block:: bash

    bob binseg test DRIU DRIVETEST -o Evaluation_DRIU_DRIVE -w https://dl.dropboxusercontent.com/s/rggn9ebj38c06uf/DRIU_DRIVE.pth
    bob binseg test M2UNet DRIVETEST -o Evaluation_M2UNet_DRIVE -w https://dl.dropboxusercontent.com/s/55xply8jm0g2skp/M2UNet_DRIVE.pth



