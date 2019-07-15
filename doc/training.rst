.. -*- coding: utf-8 -*-
.. _bob.ip.binseg.training:


========
Training
========

To replicate our results use ``bob binseg train`` followed by the model config
and the dataset config. Use ``bob binseg train --help`` for more information.

.. note::

   We strongly advice training with a GPU (using ``-d cuda``). Depending on the available GPU
   memory you might have to adjust your batch size (``-b``).

Default Dataset configs
=======================

1. Vessel:

* CHASEDB1
* CHASEDB1TEST
* COVD-DRIVE
* COVD-DRIVE_SSL
* COVD-STARE
* COVD-STARE_SSL
* COVD-IOSTARVESSEL
* COVD-IOSTARVESSEL_SSL
* COVD-HRF
* COVD-HRF_SSL
* COVD-CHASEDB1
* COVD-CHASEDB1_SSL
* DRIVE
* DRIVETEST
* HRF
* HRFTEST
* IOSTARVESSEL
* IOSTARVESSELTEST
* STARE
* STARETEST

2. Optic Disc and Cup

* DRIONSDB
* DRIONSDBTEST
* DRISHTIGS1OD
* DRISHTIGS1ODTEST
* DRISHTIGS1CUP
* DRISHTIGS1CUPTEST
* IOSTAROD
* IOSTARODTEST
* REFUGECUP
* REFUGECUPTEST
* REFUGEOD
* REFUGEODTEST
* RIMONER3CUP
* RIMONER3CUPTEST
* RIMONER3OD
* RIMONER3ODTEST

Default Model configs
=====================

* DRIU
* DRIUBN
* DRIUSSL
* DRIUBNSSL
* DRIUOD
* HED
* M2UNet
* M2UNetSSL
* UNet


Baseline Benchmarks
===================

.. code-block:: bash

    #!/bin/bash
    # set output directory
    outputroot=`pwd`"/output"
    mkdir -p $outputroot

    #### Global config ####
    m2u=M2UNet
    hed=HED
    driu=DRIU
    unet=UNet
    m2ussl=M2UNetSSL
    driussl=DRIUSSL

    #### CHASE_DB 1 ####
    dataset=CHASEDB1
    output=$outputroot"/"$dataset
    mkdir -p $output
    # batch sizes
    b_m2u=6
    b_hed=4
    b_driu=4
    b_unet=2
    # Train
    bob binseg train $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv
    bob binseg train $hed $dataset -b $b_hed -d cuda -o $output"/"$hed -vv
    bob binseg train $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg train $unet $dataset -b $b_unet -d cuda -o $output"/"$unet -vv

    #### DRIVE ####
    dataset=DRIVE
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    b_m2u=16
    b_hed=8
    b_driu=8
    b_unet=4
    # Train
    bob binseg train $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv
    bob binseg train $hed $dataset -b $b_hed -d cuda -o $output"/"$hed -vv
    bob binseg train $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg train $unet $dataset -b $b_unet -d cuda -o $output"/"$unet -vv

    #### HRF ####
    dataset=HRF
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    b_m2u=1
    b_hed=1
    b_driu=1
    b_unet=1
    # Train
    bob binseg train $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv
    bob binseg train $hed $dataset -b $b_hed -d cuda -o $output"/"$hed -vv
    bob binseg train $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg train $unet $dataset -b $b_unet -d cuda -o $output"/"$unet -vv

    #### IOSTAR VESSEL ####
    dataset=IOSTARVESSEL
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    b_m2u=6
    b_hed=4
    b_driu=4
    b_unet=2
    # Train
    bob binseg train $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv
    bob binseg train $hed $dataset -b $b_hed -d cuda -o $output"/"$hed -vv
    bob binseg train $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg train $unet $dataset -b $b_unet -d cuda -o $output"/"$unet -vv

    #### STARE ####
    dataset=STARE
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    b_m2u=6
    b_hed=4
    b_driu=5
    b_unet=2
    # Train
    bob binseg train $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv
    bob binseg train $hed $dataset -b $b_hed -d cuda -o $output"/"$hed -vv
    bob binseg train $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg train $unet $dataset -b $b_unet -d cuda -o $output"/"$unet -vv


Combined Vessel Dataset (COVD) and Semi-Supervised Learning (SSL)
=================================================================

COVD-:

.. code-block:: bash

    ### COVD-DRIVE ####
    dataset=COVD-DRIVE
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    driu=DRIU
    m2u=M2UNet
    b_driu=4
    b_m2u=8
    # Train
    bob binseg train $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg train $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv

    ### COVD-STARE ####
    dataset=COVD-STARE
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    driu=DRIU
    m2u=M2UNet
    b_driu=4
    b_m2u=4
    # Train
    bob binseg train $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg train $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv

    ### COVD-IOSTAR ####
    dataset=COVD-IOSTARVESSEL
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    driu=DRIU
    m2u=M2UNet
    b_driu=2
    b_m2u=4
    # Train
    bob binseg train $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg train $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv

    ### COVD-CHASEDB1 ####
    dataset=COVD-CHASEDB1
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    driu=DRIU
    m2u=M2UNet
    b_driu=2
    b_m2u=4
    # Train
    bob binseg train $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg train $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv

    ### COVD-HRF ####
    dataset=COVD-HRF
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    driu=DRIU
    m2u=M2UNet
    b_driu=2
    b_m2u=4
    # Train
    bob binseg train $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg train $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv


COVD-SSL:

.. code-block:: bash

    ### COVD-DRIVE_SSL ####
    dataset=COVD-DRIVE_SSL
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    driu=DRIUSSL
    m2u=M2UNetSSL
    b_driu=4
    b_m2u=4
    # Train
    bob binseg ssltrain $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg ssltrain $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv

    ### COVD-STARE_SSL ####
    dataset=COVD-STARE_SSL
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    driu=DRIUSSL
    m2u=M2UNetSSL
    b_driu=4
    b_m2u=4
    # Train
    bob binseg ssltrain $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg ssltrain $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv

    ### COVD-IOSTAR_SSL ####
    dataset=COVD-IOSTARVESSEL_SSL
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    driu=DRIUSSL
    m2u=M2UNetSSL
    b_driu=1
    b_m2u=2
    # Train
    bob binseg ssltrain $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg ssltrain $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv

    ### COVD-CHASEDB1_SSL ####
    dataset=COVD-CHASEDB1_SSL
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    driu=DRIUSSL
    m2u=M2UNetSSL
    b_driu=2
    b_m2u=2
    # Train
    bob binseg ssltrain $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg ssltrain $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv


    ### COVD-HRF_SSL ####
    dataset=COVD-HRF_SSL
    output=$outputroot"/"$dataset
    mkdir -p $output
    # model configs
    driu=DRIUSSL
    m2u=M2UNetSSL
    b_driu=1
    b_m2u=2
    # Train
    bob binseg ssltrain $driu $dataset -b $b_driu -d cuda -o $output"/"$driu -vv
    bob binseg ssltrain $m2u $dataset -b $b_m2u -d cuda -o $output"/"$m2u -vv

Using your own configs
======================

Instead of the default configs you can pass the full path of your
customized dataset and model config (both in PyTorch format).
The default configs are stored under ``bob.ip.binseg/bob/ip/binseg/configs/``.

.. code-block:: bash

    bob binseg train /path/to/model/config.py /path/to/dataset/config.py



