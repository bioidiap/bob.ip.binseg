#!/usr/bin/env python
# coding=utf-8


"""Common utilities"""


import torch
import torch.utils.data

from .transforms import Compose, ToTensor


class SampleList2TorchDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper around Sample lists

    A transform object can be passed that will be applied to the image, ground
    truth and mask (if present).

    It supports indexing such that dataset[i] can be used to get ith sample.

    Parameters
    ----------
    samples : list
        A list of :py:class:`bob.ip.binseg.data.sample.Sample` objects

    transforms : :py:class:`list`, Optional
        a list of transformations to be applied to **both** image and
        ground-truth data.  Notice that image changing transformations such as
        :py:class:`.transforms.ColorJitter` are only applied to the image and
        **not** to ground-truth.  Also notice a last transform
        (:py:class:`bob.ip.binseg.data.transforms.ToTensor`) is always applied.

    """

    def __init__(self, samples, transforms=[]):

        self._samples = samples
        self._transform = Compose(transforms + [ToTensor()])

    def __len__(self):
        """

        Returns
        -------

        size : int
            size of the dataset

        """
        return len(self._samples)

    def __getitem__(self, index):
        """

        Parameters
        ----------

        index : int

        Returns
        -------

        sample : list
            The sample data: ``[key, image[, gt[, mask]]]``

        """

        item = self._samples[index]
        data = item.data  # triggers data loading

        retval = [data["data"]]
        if "label" in data: retval.append(data["label"])
        if "mask" in data: retval.append(data["mask"])

        if self._transform:
            retval = self._transform(*retval)

        return [item.key] + retval


class SSLDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper around labelled and unlabelled sample lists

    Yields elements of the form:

    .. code-block:: text

       [key, image, ground-truth, [mask,] unlabelled-key, unlabelled-image]

    The size of the dataset is the same as the labelled dataset.

    Indexing works by selecting the right element on the labelled dataset, and
    randomly picking another one from the unlabelled dataset

    Parameters
    ----------

    labelled : :py:class:`torch.utils.data.Dataset`
        Labelled dataset (**must** have "mask" and "label" entries for every
        sample)

    unlabelled : :py:class:`torch.utils.data.Dataset`
        Unlabelled dataset (**may** have "mask" and "label" entries for every
        sample, but are ignored)

    """

    def __init__(self, labelled, unlabelled):
        self.labelled = labelled
        self.unlabelled = unlabelled

    def __len__(self):
        """

        Returns
        -------

        size : int
            size of the dataset

        """

        return len(self.labelled)

    def __getitem__(self, index):
        """

        Parameters
        ----------
        index : int
            The index for the element to pick

        Returns
        -------

        sample : list
            The sample data: ``[key, image, gt, [mask, ]unlab-key, unlab-image]``

        """

        retval = self.labelled[index]
        # gets one an unlabelled sample randomly to follow the labelled sample
        unlab = self.unlabelled[torch.randint(len(self.unlabelled))]
        # only interested in key and data
        return retval + unlab[:2]
