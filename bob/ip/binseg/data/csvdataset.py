#!/usr/bin/env python
# coding=utf-8

import os
import csv

from PIL import Image

from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as VF

import bob.io.base


class CSVDataset(Dataset):
    """
    Generic filelist dataset

    To create a new dataset, you only need to provide a CSV formatted filelist
    using any separator (e.g. comma, space, semi-colon) including, in the first
    column, a path pointing to the input image, and in the second column, a
    path pointing to the ground truth.  Relative paths are interpreted with
    respect to the location where the CSV file is or to an optional
    ``root_path`` parameter, that must be also provided.

    There are no requirements concerning image or ground-truth homogenity.
    Anything that can be loaded by our image and data loaders is OK.  Use
    a non-white character as separator.  Here is a far too complicated example:

    .. code-block:: text

       /path/to/image1.jpg,/path/to/ground-truth1.png
       /possibly/another/path/to/image 2.PNG,/path/to/that/ground-truth.JPG
       relative/path/image3.gif,relative/path/gt3.gif

    .. important::

       Images are converted to RGB after readout via PIL.  Ground-truth data is
       loaded using the same technique, but converted to mode ``1`` instead of
       ``RGB``.  If ground-truth data is encoded as an HDF5 file, we use
       instead :py:func:`bob.io.base.load`, and then converted it to 32-bit
       float data.

    To generate a dataset without ground-truth (e.g. for prediction tasks),
    then omit the second column.


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


    def has_ground_truth(self):
        """Tells if this dataset has ground-truth or not"""
        return len(self.data[0]) > 1


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
            ``[name, img, gt]`` or ``[name, img]`` depending on whether this
            dataset has or not ground-truth.
        """

        sample_paths = self.data[index]

        img_path = sample_paths[0]
        gt_path = sample_paths[1] if len(sample_paths) > 1 else None

        # images are converted to RGB mode automatically
        sample = [Image.open(img_path).convert(mode="RGB")]

        if gt_path is not None:
            if gt_path.endswith(".hdf5"):
                gt = bob.io.base.load(str(gt_path)).astype("float32")
                # a bit hackish, but will get what we need
                gt = VF.to_pil_image(torch.from_numpy(gt))
            else:
                gt = Image.open(gt_path)
            gt = gt.convert(mode="1", dither=None)
            sample = sample + [gt]

        if self.transform:
            sample = self.transform(*sample)

        return [img_path] + sample
