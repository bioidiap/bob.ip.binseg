#!/usr/bin/env python
# coding=utf-8

from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset

from .transforms import Compose, ToTensor


def _find_files(data_path, glob):
    """
    Recursively retrieves file lists from a given path, matching a given glob

    This function will use :py:meth:`pathlib.Path.rglob`, together with the
    provided glob pattern to search for anything the desired filename.
    """

    data_path = Path(data_path)
    return sorted(list(data_path.rglob(glob)))


class FolderDataset(Dataset):
    """
    Generic image folder containing images for prediction

    .. important::

        This implementation, contrary to its sister
        :py:class:`.csvdataset.CSVDataset`, does not *automatically* convert
        the input image to RGB, before passing it to the transforms, so it is
        possible to accomodate a wider range of input types (e.g. 16-bit PNG
        images).

    Parameters
    ----------

    path : str
        full path to root of dataset

    glob : str
        glob that can be used to filter-down files to be loaded on the provided
        path

    transforms : :py:class:`list`, Optional
        a list of transformations to be applied to **both** image and
        ground-truth data.  Notice that image changing transformations such as
        :py:class:`.transforms.ColorJitter` are only applied to the image and
        **not** to ground-truth.  Also notice a last transform
        (:py:class:`bob.ip.binseg.data.transforms.ToTensor`) is always applied.

    """

    def __init__(self, path, glob="*", transforms=[]):
        self.transform = Compose(transforms + [ToTensor()])
        self.path = path
        self.data = _find_files(path, glob)

    def __len__(self):
        """
        Returns
        -------
        int
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
            [name, img]
        """

        sample = [Image.open(self.data[index])]
        if self.transform:
            sample = self.transform(*sample)
        return [self.data[index].relative_to(self.path).as_posix()] + sample
