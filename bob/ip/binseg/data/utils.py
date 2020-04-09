#!/usr/bin/env python
# coding=utf-8


"""Common utilities"""


import functools

import nose.plugins.skip

import torch
import torch.utils.data

import bob.extension


def rc_variable_set(name):
    """
    Decorator that checks if a given bobrc variable is set before running
    """

    def wrapped_function(test):
        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            if name not in bob.extension.rc:
                raise nose.plugins.skip.SkipTest("Bob's RC variable '%s' is not set" % name)
            return test(*args, **kwargs)

        return wrapper

    return wrapped_function


class DelayedSample2TorchDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper around DelayedSample lists

    A transform object can be passed that will be applied to the image, ground
    truth and mask (if present).

    It supports indexing such that dataset[i] can be used to get ith sample.

    Parameters
    ----------
    samples : list
        A list of :py:class:`bob.ip.binseg.data.sample.DelayedSample` objects

    transform : :py:mod:`bob.ip.binseg.data.transforms`, optional
        A transform or composition of transfroms. Defaults to ``None``.

    """

    def __init__(self, samples, transform=None):

        self._samples = samples
        self._transform = transform

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
