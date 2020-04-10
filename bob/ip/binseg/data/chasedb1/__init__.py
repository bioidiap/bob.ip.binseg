#!/usr/bin/env python
# coding=utf-8

"""CHASE-DB1 dataset for Vessel Segmentation

The CHASE_DB1 is a retinal vessel reference dataset acquired from multiethnic
school children. This database is a part of the Child Heart and Health Study in
England (CHASE), a cardiovascular health survey in 200 primary schools in
London, Birmingham, and Leicester. The ocular imaging was carried out in
46 schools and demonstrated associations between retinal vessel tortuosity and
early risk factors for cardiovascular disease in over 1000 British primary
school children of different ethnic origin. The retinal images of both of the
eyes of each child were recorded with a hand-held Nidek NM-200-D fundus camera.
The images were captured at 30 degrees FOV camera. The dataset of images are
characterized by having nonuniform back-ground illumination, poor contrast of
blood vessels as compared with the background and wider arteriolars that have a
bright strip running down the centre known as the central vessel reflex.

* Reference: [CHASEDB1-2012]_
* Original resolution (height x width): 960 x 999
* Split reference: [CHASEDB1-2012]_
* Protocol ``default``:

  * Training samples: 8 (including labels from annotator "1stHO")
  * Test samples: 20 (including labels from annotator "1stHO")

* Protocol ``second-annotation``:

  * Training samples: 8 (including labels from annotator "2ndHO")
  * Test samples: 20 (including labels from annotator "2ndHO")

"""

import os
import pkg_resources

import bob.extension

from ..jsondataset import JSONDataset
from ..loader import load_pil_rgb, load_pil_1

_protocols = [
        pkg_resources.resource_filename(__name__, "default.json"),
        pkg_resources.resource_filename(__name__, "second-annotation.json"),
        ]

_root_path = bob.extension.rc.get('bob.ip.binseg.chasedb1.datadir',
        os.path.realpath(os.curdir))

def _loader(s):
    return dict(
            data=load_pil_rgb(s["data"]),
            label=load_pil_1(s["label"]),
            )

dataset = JSONDataset(protocols=_protocols, root_path=_root_path, loader=_loader)
"""CHASE-DB1 dataset object"""
