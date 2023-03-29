.. SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. place re-used URLs here, then include this file
.. on your other RST sources.


.. _idiap: http://www.idiap.ch
.. _pytorch: https://pytorch.org
.. _tabulate: https://pypi.org/project/tabulate/
.. _our paper: https://arxiv.org/abs/1909.03856
.. _cla: https://en.wikipedia.org/wiki/Contributor_License_Agreement
.. _project harmony: http://www.harmonyagreements.org/
.. _tto: mailto:tto@idiap.ch
.. _pip: https://pip.pypa.io/en/stable/
.. _mamba: https://mamba.readthedocs.io/en/latest/index.html

.. Raw data websites
.. _drive: https://github.com/wfdubowen/Retina-Unet/tree/master/DRIVE/
.. _stare: http://cecas.clemson.edu/~ahoover/stare/
.. _hrf: https://www5.cs.fau.de/research/data/fundus-images/
.. _iostar: http://www.retinacheck.org/datasets
.. _chase-db1: https://blogs.kingston.ac.uk/retinal/chasedb1/
.. _drions-db: http://www.ia.uned.es/~ejcarmona/DRIONS-DB.html
.. _rim-one r3: http://medimrg.webs.ull.es/research/downloads/
.. _drishti-gs1: http://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php
.. _refuge: https://refuge.grand-challenge.org/Details/
.. _amd grand-challenge: https://amd.grand-challenge.org/
.. _drhagis: https://personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/
.. _montgomery county: https://openi.nlm.nih.gov/faq#faq-tb-coll
.. _jsrt: http://db.jsrt.or.jp/eng.php
.. _jsrt-kaggle: https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt
.. _shenzhen: https://openi.nlm.nih.gov/faq#faq-tb-coll
.. _cxr8: https://nihcc.app.box.com/v/ChestXray-NIHCC

.. Annotation data websites
.. _shenzhen annotations: https://www.kaggle.com/yoctoman/shcxr-lung-mask
.. _cxr8 annotations: https://github.com/lucasmansilla/NIH_chest_xray14_segmentations
.. _jsrt annotations: https://www.isi.uu.nl/Research/Databases/SCR/download.php


.. Software Tools
.. _maskrcnn-benchmark: https://github.com/facebookresearch/maskrcnn-benchmark

.. _wilcoxon test: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test


.. Pretrained models

.. _baselines_driu_drive: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/driu-drive-1947d9fa.pth
.. _baselines_hed_drive: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/hed-drive-c8b86082.pth
.. _baselines_m2unet_drive: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/m2unet-drive-ce4c7a53.pth
.. _baselines_unet_drive: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/unet-drive-0ac99e2e.pth
.. _baselines_driu_stare: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/driu-stare-79dec93a.pth
.. _baselines_hed_stare: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/hed-stare-fcdb7671.pth
.. _baselines_m2unet_stare: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/m2unet-stare-952778c2.pth
.. _baselines_unet_stare: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/unet-stare-49b6a6d0.pth
.. _baselines_driu_chase: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/driu-chasedb1-e7cf53c3.pth
.. _baselines_hed_chase: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/hed-chasedb1-55ec6d34.pth
.. _baselines_m2unet_chase: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/m2unet-chasedb1-0becbf29.pth
.. _baselines_unet_chase: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/unet-chasedb1-be41b5a5.pth
.. _baselines_driu_hrf: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/driu-hrf-c9e6a889.pth
.. _baselines_hed_hrf: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/hed-hrf-3f4ab1c4.pth
.. _baselines_m2unet_hrf: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/m2unet-hrf-2c3f2485.pth
.. _baselines_unet_hrf: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/unet-hrf-9a559821.pth
.. _baselines_driu_iostar: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/driu-iostar-vessel-ef8cc27b.pth
.. _baselines_hed_iostar: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/hed-iostar-vessel-37cfaee1.pth
.. _baselines_m2unet_iostar: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/m2unet-iostar-vessel-223b61ef.pth
.. _baselines_unet_iostar: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/baselines/unet-iostar-vessel-86c78e87.pth

.. _baselines_m2unet_jsrt: https://bobconda.lab.idiap.ch/public/data/bob/deepdraw/master/baselines/m2unet-jsrt-5f062009.pth
.. _baselines_m2unet_montgomery: https://bobconda.lab.idiap.ch/public/data/bob/deepdraw/master/baselines/m2unet-montgomery-1c24519a.pth
.. _baselines_m2unet_shenzhen: https://bobconda.lab.idiap.ch/public/data/bob/deepdraw/master/baselines/m2unet-shenzhen-7c9688e6.pth
.. _baselines_lwnet_jsrt: https://bobconda.lab.idiap.ch/public/data/bob/deepdraw/master/baselines/lwnet-jsrt-73807eb1.pth
.. _baselines_lwnet_montgomery: https://bobconda.lab.idiap.ch/public/data/bob/deepdraw/master/baselines/lwnet-montgomery-9c6bf39b.pth
.. _baselines_lwnet_shenzhen: https://bobconda.lab.idiap.ch/public/data/bob/deepdraw/master/baselines/lwnet-shenzhen-10196d9c.pth

.. _covd_driu_drive: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/driu/drive/model.pth
.. _covd_hed_drive: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/hed/drive/model.pth
.. _covd_m2unet_drive: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/m2unet/drive/model.pth
.. _covd_unet_drive: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/unet/drive/model.pth
.. _covd_driu_stare: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/driu/stare/model.pth
.. _covd_hed_stare: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/hed/stare/model.pth
.. _covd_m2unet_stare: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/m2unet/stare/model.pth
.. _covd_unet_stare: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/unet/stare/model.pth
.. _covd_driu_chase: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/driu/chasedb1/model.pth
.. _covd_hed_chase: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/hed/chasedb1/model.pth
.. _covd_m2unet_chase: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/m2unet/chasedb1/model.pth
.. _covd_unet_chase: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/unet/chasedb1/model.pth
.. _covd_driu_hrf: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/driu/hrf/model.pth
.. _covd_hed_hrf: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/hed/hrf/model.pth
.. _covd_m2unet_hrf: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/m2unet/hrf/model.pth
.. _covd_unet_hrf: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/unet/hrf/model.pth
.. _covd_driu_iostar: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/driu/iostar-vessel/model.pth
.. _covd_hed_iostar: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/hed/iostar-vessel/model.pth
.. _covd_m2unet_iostar: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/m2unet/iostar-vessel/model.pth
.. _covd_unet_iostar: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/covd/unet/iostar-vessel/model.pth

.. DRIVE
.. _driu_drive.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/DRIU_DRIVE.pth
.. _m2unet_drive.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_DRIVE.pth
.. _m2unet_covd-drive.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_COVD-DRIVE.pth
.. _m2unet_covd-drive_ssl.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_COVD-DRIVE_SSL.pth

.. STARE
.. _driu_stare.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/DRIU_STARE.pth
.. _m2unet_stare.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_STARE.pth
.. _m2unet_covd-stare.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_COVD-STARE.pth
.. _m2unet_covd-stare_ssl.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_COVD-STARE_SSL.pth

.. CHASE-DB1
.. _driu_chasedb1.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/DRIU_CHASEDB1.pth
.. _m2unet_chasedb1.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_CHASEDB1.pth
.. _m2unet_covd-chasedb1.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_COVD-CHASEDB1.pth
.. _m2unet_covd-chasedb1_ssl.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_COVD-CHASEDB1_SSL.pth

.. IOSTAR
.. _driu_iostar.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/DRIU_IOSTARVESSEL.pth
.. _m2unet_iostar.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_IOSTARVESSEL.pth
.. _m2unet_covd-iostar.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_COVD-IOSTAR.pth
.. _m2unet_covd-iostar_ssl.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_COVD-IOSTAR_SSL.pth

.. HRF
.. _driu_hrf.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/DRIU_HRF1168.pth
.. _m2unet_hrf.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_HRF1168.pth
.. _m2unet_covd-hrf.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_COVD-HRF.pth
.. _m2unet_covd-hrf_ssl.pth: https://www.idiap.ch/software/bob/data/bob/deepdraw/master/M2UNet_COVD-HRF_SSL.pth
