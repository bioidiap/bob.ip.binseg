#!/usr/bin/env python
# coding=utf-8

import os
import csv

from PIL import Image

from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as VF

import bob.io.base

import logging
logger = logging.getLogger(__name__)


class CSVDataset(Dataset):
    """
    Generic filelist dataset

    To create a new dataset, you only need to provide a CSV formatted filelist
    using any separator (e.g. comma, space, semi-colon) with the following
    information:

    .. code-block:: text

       image[,label[,mask]]

    Where:

    * ``image``: absolute or relative path leading to original image
    * ``label``: (optional) absolute or relative path with manual segmentation
      information
    * ``mask``: (optional) absolute or relative path with a mask that indicates
      valid regions in the image where automatic segmentation should occur

    Relative paths are interpreted with respect to the location where the CSV
    file is or to an optional ``root_path`` parameter, that can be provided.

    There are no requirements concerning image or ground-truth homogenity.
    Anything that can be loaded by our image and data loaders is OK.  Use
    a non-white character as separator.  Example

    .. code-block:: text

       image1.jpg,gt1.tif,mask1.png
       image2.png,gt2.png,mask2.png
       ...


    Notice that all rows must have the same number of entries.

    .. important::

       Images are converted to RGB after readout via PIL.  Ground-truth data is
       loaded using the same technique, but converted to mode ``1`` instead of
       ``RGB``.  If ground-truth data is encoded as an HDF5 file, we use
       instead :py:func:`bob.io.base.load`, and then converted it to 32-bit
       float data.

    To generate a dataset without ground-truth (e.g. for prediction tasks),
    then omit the second and third columns.


    Parameters
    ----------
    path : str
        Full path to the file containing the dataset description, in CSV
        format as described above

    root_path : :py:class:`str`, Optional
        Path to a common filesystem root where files with relative paths should
        be sitting.  If not set, then we use the absolute path leading to the
        CSV file as ``root_path``

    check_available : :py:class:`bool`, Optional
        If set to ``True``, then checks if files in the file list are
        available.  Otherwise does not.

    transform : :py:class:`.transforms.Compose`, Optional
        a composition of transformations to be applied to **both** image and
        ground-truth data.  Notice that image changing transformations such as
        :py:class:`.transforms.ColorJitter` are only applied to the image and
        **not** to ground-truth.

    """

    def __init__(self, path, root_path=None, check_available=True, transform=None):

        self.root_path = root_path or os.path.dirname(path)
        self.transform = transform

        def _make_abs_path(root, s):
            retval = []
            for p in s:
                if not os.path.isabs(p):
                    retval.append(os.path.join(root, p))
            return retval

        with open(path, newline='') as f:
            reader = csv.reader(f)
            self.data = [_make_abs_path(self.root_path, k) for k in reader]

        # check if all files are readable, warn otherwise
        if check_available:
            errors = 0
            for s in self.data:
                for p in s:
                    if not os.path.exists(p):
                        errors += 1
                        logger.error(f"Cannot find {p}")
            assert errors == 0, f"There {errors} files which cannot be " \
                    f"found on your filelist ({path}) dataset"

        # check all data entries have the same size
        assert all(len(k) == len(self.data[0]) for k in self.data), \
                f"There is an inconsistence on your dataset - not all " \
                f"entries have length=={len(self.data[0])}"


    def __len__(self):
        """

        Returns
        -------

        length : int
            size of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """

        Parameters
        ----------
        index : int

        Returns
        -------
        sample : list
            ``[name, img, gt, mask]``, ``[name, img, gt]`` or ``[name, img]``
            depending on whether this dataset has or not ground-truth
            annotations and masks.  The value of ``name`` is relative to
            ``root_path``, in cases it starts with ``root_path``.
        """

        sample_paths = self.data[index]

        img_path = sample_paths[0]
        meta_data = sample_paths[1:]

        # images are converted to RGB mode automatically
        sample = [Image.open(img_path).convert(mode="RGB")]

        # ground-truth annotations and masks are treated the same
        for path in meta_data:
            if path is not None:
                if path.endswith(".hdf5"):
                    data = bob.io.base.load(str(path)).astype("float32")
                    # a bit hackish, but will get what we need
                    data = VF.to_pil_image(torch.from_numpy(data))
                else:
                    data = Image.open(path)
                sample += [data.convert(mode="1", dither=None)]

        if self.transform:
            sample = self.transform(*sample)

        # make paths relative if necessary
        stem = img_path
        if stem.startswith(self.root_path):
            stem = os.path.relpath(stem, self.root_path)
        elif stem.startswith(os.pathsep):
            stem = stem[len(os.pathsep):]

        return [stem] + sample
