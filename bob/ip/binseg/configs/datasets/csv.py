#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Example CSV-based filelist dataset

In case you have your own dataset that is organized on your filesystem, this
configuration shows an example setup so you can feed such files and
ground-truth data to train one of the available network models or to evaluate
it.

You must write CSV based file (e.g. using comma as separator) that describes
the image and ground-truth locations for each image pair on your dataset.
Relative paths are considered with respect to the location of the CSV file
itself by default, also pass the ``root_path`` parameter to the
:py:class:`bob.ip.binseg.data.csvdataset.CSVDataset` object constructor.  So,
for example, if you have a structure like this:

.. code-block:: text

   ├── images
       ├── image_1.png
       ├── ...
       └── image_n.png
   └── ground-truth
       ├── gt_1.png
       ├── ...
       └── gt_n.png

Then create a file in the same level of ``images`` and ``ground-truth`` with
the following contents:

.. code-block:: text

   images/image_1.png,ground-truth/gt_1.png
   ...,...
   images/image_n.png,ground-truth/gt_n.png

To create a dataset without ground-truth (e.g., for prediction purposes), then
omit the second column on the CSV file.

Use the path leading to the CSV file and replace ``<path.csv>`` on the example
code for this configuration, that you must copy locally to make changes:

.. code-block:: sh

   $ bob binseg config copy csv-dataset-example mydataset.py
   # edit mydataset.py as explained here

Fine-tune the transformations for your particular purpose:

    1. If you are training a new model, you may add random image
       transformations
    2. If you are running prediction, you should/may skip random image
       transformations

Keep in mind that specific models require that you feed images respecting
certain restrictions (input dimensions, image centering, etc.).  Check the
configuration that was used to train models and try to match it as well as
possible.

See:

* :py:class:`bob.ip.binseg.data.csvdataset.CSVDataset` for operational details.
* :py:class:`bob.ip.binseg.data.folderdataset.FolderDataset` for an alternative
   implementation of an easier to generate **prediction** dataset.

"""

# add your transforms below - these are just examples
from bob.ip.binseg.data.transforms import CenterCrop
#from bob.ip.binseg.configs.datasets.utils import DATA_AUGMENTATION as _DA
_transforms = [
        CenterCrop((544, 544)),
        ] # + _DA

from bob.ip.binseg.data.csvdataset import CSVDataset
#dataset = CSVDataset("<path.csv>", check_available=False, transforms=_transforms)
