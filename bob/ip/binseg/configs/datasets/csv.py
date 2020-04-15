#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Example CSV-based filelist dataset

In case you have your own dataset that is organized on your filesystem, this
configuration shows an example setup so you can feed such files (and
ground-truth data) to train one of the available network models or to evaluate
it.

You must write CSV based file (e.g. using comma as separator) that describes
the image and ground-truth locations for each image pair on your dataset.  So,
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

Then create a file with the following contents:

.. code-block:: text

   images/image_1.png,ground-truth/gt_1.png
   ...,...
   images/image_n.png,ground-truth/gt_n.png

To create a dataset without ground-truth (e.g., for prediction purposes), then
omit the second column on the CSV file.

Use the path leading to the CSV file and carefully read the comments in this
configuration.  **Copy it locally to make changes**:

.. code-block:: sh

   $ bob binseg config copy csv-dataset-example mydataset.py
   # edit mydataset.py as explained here, follow the comments

Fine-tune the transformations for your particular purpose:

    1. If you are training a new model, you may add random image
       transformations
    2. If you are running prediction, you should/may skip random image
       transformations

Keep in mind that specific models require that you feed images respecting
certain restrictions (input dimensions, image centering, etc.).  Check the
configuration that was used to train models and try to match it as well as
possible.

Finally, you must create a connector that will act as a "dataset" for pytorch.
The connector make a list of samples, returned by your raw dataset, look like
something our pytorch setup can digest (tuples of data with a certain
organisation).

More information:

* :py:class:`bob.ip.binseg.data.dataset.CSVDataset` for operational details.
* :py:class:`bob.ip.binseg.data.dataset.JSONDataset` for an alternative for
  multi-protocol datasets (all of our supported raw datasets are implemented
  using this)
* :py:class:`bob.ip.binseg.data.utils.SampleList2TorchDataset` for extra
  information on the sample list to pytorch connector

"""

# First, define how to access and load the raw data. Our package provides some
# stock loaders we use for other datasets. You may have a look at the
# documentation of that module for details.
from bob.ip.binseg.data.loaders import (
    load_pil_rgb,
    load_pil_1,
    data_path_keymaker,
)

# How we use the loaders - "sample" is a dictionary where keys are defined
# below and map to the columns of the CSV files you input.
def _loader(context, sample):
    # "context" is ignored in this case - database is homogeneous
    # it is a dictionary that passes e.g., the name of the subset
    # being loaded, so you can take contextual decisions on the loading

    # Using the path leading to the various data files stored in disk allows
    # the CSV file to contain only relative paths and is, therefore, more
    # compact.  Of course, you can make those paths absolute and then simplify
    # it here.
    import os
    root_path = "/path/where/raw/files/sit"

    return dict(
        data=load_pil_rgb(os.path.join(root_path, sample["data"])),
        label=load_pil_1(os.path.join(root_path, sample["label"])),
    )

# This is just a class that puts everything together: the CSV file, how to load
# each sample defined in the dataset, names for the various columns of the CSV
# file and how to make unique keys for each sample (keymaker).  Once created,
# this object can be called to generate sample lists.
from bob.ip.binseg.data.dataset import CSVDataset
raw_dataset = CSVDataset(
    # path to the CSV file(s) - you may add as many subsets as you want, each
    # with an unique name, you'll use later to generate sample lists
    subsets=dict(data="<path/to/train.csv>"),
    fieldnames=("data", "label"),  #these are the column names
    loader=_loader,
    keymaker=data_path_keymaker,
)

# Finally, we build a connector to passes our dataset to the pytorch framework
# so we can, for example, evaluate a trained pytorch model

# Add/tune your transforms below - these are just examples compatible with a
# model that requires image inputs of 544 x 544 pixels.
from bob.ip.binseg.data.transforms import CenterCrop

# from bob.ip.binseg.configs.datasets.utils import DATA_AUGMENTATION as _DA
_transforms = [
    CenterCrop((544, 544)),
]  # + _DA

# This class will simply trigger data loading and re-arrange the data so that
# data is fed in the right order to pytorch: (key, image[, label[, mask]]).
# This class also inherits from pytorch Dataset and respect its required API.
# See the documentation for details.
from bob.ip.binseg.data.utils import SampleList2TorchDataset
dataset = SampleList2TorchDataset(raw_dataset.subset("data"), _transforms)
