#!/usr/bin/env python
# coding=utf-8

"""Example CSV-based custom filelist dataset

In case you have your own dataset that is organized on your filesystem (or
elsewhere), this configuration shows an example setup so you can feed such data
(potentially including any ground-truth you may have) to train, predict or
evaluate one of the available network models.

You must write CSV based file (e.g. using comma as separator) that describes
the data (and ground-truth) locations for each sample on your dataset.  So, for
example, if you have a file structure like this:

.. code-block:: text

   ├── images
       ├── image_1.png
       ├── ...
       └── image_n.png
   └── ground-truth
       ├── gt_1.png
       ├── ...
       └── gt_n.png

Then create one or more files, each containing a subset of your dataset:

.. code-block:: text

   images/image_1.png,ground-truth/gt_1.png
   ...,...
   images/image_n.png,ground-truth/gt_n.png

To create a subset without ground-truth (e.g., for prediction purposes), then
omit the second column on the CSV file.

Use the path leading to the CSV file and carefully read the comments in this
configuration.  **Copy it locally to make changes**:

.. code-block:: sh

   $ bob binseg config copy csv-dataset-example mydataset.py
   # edit mydataset.py as explained here, follow the comments

Finally, the only object this file needs to provide is one named ``dataset``,
and it should contain a dictionary mapping a name, such as ``train``, ``dev``,
or ``test``, to objects of type :py:class:`torch.utils.data.Dataset`.  As you
will see in this example, we provide boilerplate code to do so.

More information:

* :py:class:`bob.ip.binseg.data.dataset.CSVDataset` for operational details.
* :py:class:`bob.ip.binseg.data.dataset.JSONDataset` for an alternative for
  multi-protocol datasets (all of our supported raw datasets are implemented
  using this)
* :py:func:`bob.ip.binseg.configs.datasets.make_dataset` for extra
  information on the sample list to pytorch connector.

"""

import os

# First, define how to access and load the raw data. Our package provides some
# stock loaders we use for other datasets. You may have a look at the
# documentation of that module for details.
from bob.ip.binseg.data.loader import (
    load_pil_rgb,
    load_pil_1,
)

from bob.ip.binseg.data.sample import Sample

# How we use the loaders - "sample" is a dictionary where keys are defined
# below and map to the columns of the CSV files you input.  This one is
# configured to load images and labels using PIL.
def _loader(context, sample):
    # "context" is ignored in this case - database is homogeneous
    # it is a dictionary that passes e.g., the name of the subset
    # being loaded, so you can take contextual decisions on the loading

    # Using the path leading to the various data files stored in disk allows
    # the CSV file to contain only relative paths and is, therefore, more
    # compact.  Of course, you can make those paths absolute and then simplify
    # it here.
    root_path = "/path/where/raw/files/sit"

    data=load_pil_rgb(os.path.join(root_path, sample["data"]))
    label=load_pil_1(os.path.join(root_path, sample["label"]))

    # You may also return DelayedSample to avoid data loading to take place
    # as the sample object itself is created.  Take a look at our own datasets
    # for examples.
    return Sample(
            key=os.path.splitext(sample["data"])[0],
            data=dict(data=data, label=label),
            )


# This is just a class that puts everything together: the CSV file, how to load
# each sample defined in the dataset, and names for the various columns of the
# CSV file.  Once created, this object can be called to generate sample lists.
from bob.ip.binseg.data.dataset import CSVDataset

_raw_dataset = CSVDataset(
    # path to the CSV file(s) - you may add as many subsets as you want:
    # * "__train__" is used for training a model (stock data augmentation is
    #    applied via our "make_dataset()" connector)
    # * anything else can be used for prediction and/or evaluation (if labels
    #   are also provided in such a set).  Data augmentation is NOT applied
    #   using our "make_dataset()" connector.
    subsets={
        "__train__": "<path/to/train.csv>",  #applies data augmentation
        "train": "<path/to/train.csv>",  #no data augmentation, evaluate it
        "test": "<path/to/test.csv>",  #no data augmentation, evaluate it
        },
    fieldnames=("data", "label"),  # these are the column names
    loader=_loader,
)

# Finally, we build a connector to passes our dataset to the pytorch framework
# so we can, for example, train and evaluate a pytorch model.  The connector
# only converts the sample lists into a standard tuple (data[, label[, mask]])
# that is expected by our engines, after applying the (optional)
# transformations you define.

# Add/tune your (optional) transforms below - these are just examples
# compatible with a model that requires image inputs of 544 x 544 pixels.
from bob.ip.binseg.data.transforms import CenterCrop
from bob.ip.binseg.configs.datasets import make_dataset as _maker

#dataset = _maker(_raw_dataset.subsets(), [CenterCrop((544, 544))])
