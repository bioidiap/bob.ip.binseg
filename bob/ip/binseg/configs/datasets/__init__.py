#!/usr/bin/env python
# coding=utf-8

"""Standard configurations for dataset setup"""


from ...data.transforms import (
    RandomRotation as _rotation,
    RandomHorizontalFlip as _hflip,
    RandomVerticalFlip as _vflip,
    ColorJitter as _jitter,
)


AUGMENTATION_ROTATION = [_rotation()]
"""Shared data augmentation based on random rotation only"""


AUGMENTATION_WITHOUT_ROTATION = [_hflip(), _vflip(), _jitter()]
"""Shared data augmentation transforms without random rotation"""


AUGMENTATION = AUGMENTATION_ROTATION + AUGMENTATION_WITHOUT_ROTATION
"""Shared data augmentation transforms"""


def make_subset(l, transforms):
    """Creates a new data set, applying transforms

    Parameters
    ----------

    l : list
        List of delayed samples

    transforms : list
        A list of transforms that needs to be applied to all samples in the set


    Returns
    -------

    subset : :py:class:`torch.utils.data.Dataset`
        A pre-formatted dataset that can be fed to one of our engines

    """

    from ...data.utils import SampleList2TorchDataset as wrapper
    return wrapper(l, transforms)


def make_trainset(l, transforms, rotation_before=False):
    """Creates a new training set, with data augmentation

    Typically, the transforms are chained to a default set of data augmentation
    operations (random rotation, horizontal and vertical flips, and color
    jitter), but flag allows prefixing the rotation specially (useful for some
    COVD training sets).


    Parameters
    ----------

    l : list
        List of delayed samples

    transforms : list
        A list of transforms that needs to be applied to all samples in the set


    Returns
    -------

    subset : :py:class:`torch.utils.data.Dataset`
        A pre-formatted dataset that can be fed to one of our engines

    """

    if rotation_before:
        return make_subset(l, AUGMENTATION_ROTATION + transforms + \
                AUGMENTATION_WITHOUT_ROTATION)

    return make_subset(l, transforms + AUGMENTATION)


def make_dataset(subsets, transforms):
    """Creates a new configuration dataset from dictionary and transforms

    This function takes as input a dictionary as those that can be returned by
    :py:meth:`bob.ip.binseg.data.dataset.JSONDataset.subsets`, mapping protocol
    names (such as ``train``, ``dev`` and ``test``) to
    :py:class:`bob.ip.binseg.data.sample.DelayedSample` lists, and a set of
    transforms, and returns a dictionary applying
    :py:class:`bob.ip.binseg.data.utils.SampleList2TorchDataset` to these
    lists, and our standard data augmentation if a ``train`` set exists.

    Parameters
    ----------

    subsets : dict
        A dictionary that contains the delayed sample lists for a number of
        named lists.  If one of the keys is ``train``, our standard dataset
        augmentation transforms are appended to the definition of that subset.
        All other subsets remain un-augmented.

    transforms : list
        A list of transforms that needs to be applied to all samples in the set


    Returns
    -------

    dataset : dict
        A pre-formatted dataset that can be fed to one of our engines. It maps
        string names to :py:class:`torch.utils.data.Dataset`'s.

    """

    retval = {}

    for key in subsets.keys():
        if key == "train":
            retval[key] = make_trainset(subsets[key], transforms)
        else:
            retval[key] = make_subset(subsets[key], transforms)

    return retval
