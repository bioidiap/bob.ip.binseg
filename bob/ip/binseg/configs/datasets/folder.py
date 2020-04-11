#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Example self-scanning folder-based dataset

In case you have data that is organized on your filesystem, this configuration
shows an example setup so you can feed such files **without** ground-truth to
predict vessel probalities using one of our trained models.  There can be any
number of images within the root folder of your dataset, with any kind of
subfolder arrangements.  For example:

.. code-block:: text

   ├── image_1.png
   └── subdir1
       ├── image_subdir_1.jpg
       ├── ...
       └── image_subdir_k.jpg
   ├── ...
   └── image_n.png

Use the path leading to the root of your dataset, and replace ``<path.csv>`` on
the example code for this configuration, that you must copy locally to make
changes:

.. code-block:: sh

   $ bob binseg config copy folder-dataset-example mydataset.py
   # edit mydataset.py as explained here

Fine-tune the transformations for your particular purpose.

Keep in mind that specific models require that you feed images respecting
certain restrictions (input dimensions, image centering, etc.).  Check the
configuration that was used to train models and try to match it as well as
possible.
"""

# add your transforms below - these are just examples
from bob.ip.binseg.data.transforms import CenterCrop
_transforms = [
        #CenterCrop((544, 544)),
    ]

from bob.ip.binseg.data.folderdataset import FolderDataset
#dataset = FolderDataset("<path.csv>", glob="*.*", transforms=_transforms)
