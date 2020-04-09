#!/usr/bin/env python
# coding=utf-8


"""Common utilities"""


import functools
import nose.plugins.skip
import torch.utils.data
import bob.extension


def rc_variable_set(name):
    """
    Decorator that checks if a given bobrc variable is set before running
    """

    def wrapped_function(test):
        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            if bob.extension.rc[name]:
                return test(*args, **kwargs)
            else:
                raise nose.plugins.skip.SkipTest("Bob's RC variable '%s' is not set" % name)

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

        self.samples = samples
        self.transform = transform

    def __len__(self):
        """

        Returns
        -------

        size : int
            size of the dataset

        """
        return len(self.samples)

    def __getitem__(self, index):
        """

        Parameters
        ----------

        index : int

        Returns
        -------

        sample : tuple
            The sample data: ``[key, image[, gt[, mask]]]``

        """

        item = self.samples[index]
        data = item.data  # triggers data loading

        retval = [data["data"]]
        if "label" in data: retval.append(data["label"])
        if "mask" in data: retval.append(data["mask"])

        if self.transform:
            retval = self.transform(*retval)

        return [item.key] + retval
