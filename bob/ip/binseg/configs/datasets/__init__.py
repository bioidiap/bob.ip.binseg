#!/usr/bin/env python
# coding=utf-8

"""Standard configurations for dataset setup"""


from ...data.transforms import (
    RandomRotation as _rotation,
    RandomHorizontalFlip as _hflip,
    RandomVerticalFlip as _vflip,
    ColorJitter as _jitter,
)


RANDOM_ROTATION = [_rotation()]
"""Shared data augmentation based on random rotation only"""


RANDOM_FLIP_JITTER = [_hflip(), _vflip(), _jitter()]
"""Shared data augmentation transforms without random rotation"""


def make_subset(l, transforms, prefixes=[], suffixes=[]):
    """Creates a new data set, applying transforms

    .. note::

       This is a convenience function for our own dataset definitions inside
       this module, guaranteeting homogenity between dataset definitions
       provided in this package.  It assumes certain strategies for data
       augmentation that may not be translatable to other applications.


    Parameters
    ----------

    l : list
        List of delayed samples

    transforms : list
        A list of transforms that needs to be applied to all samples in the set

    prefixes : list
        A list of data augmentation operations that needs to be applied
        **before** the transforms above

    suffixes : list
        A list of data augmentation operations that needs to be applied
        **after** the transforms above


    Returns
    -------

    subset : :py:class:`bob.ip.binseg.data.utils.SampleListDataset`
        A pre-formatted dataset that can be fed to one of our engines

    """

    from ...data.utils import SampleListDataset as wrapper

    return wrapper(l, prefixes + transforms + suffixes)


def augment_subset(s, rotation_before=False):
    """Creates a new subset set, **with data augmentation**

    Typically, the transforms are chained to a default set of data augmentation
    operations (random rotation, horizontal and vertical flips, and color
    jitter), but a flag allows prefixing the rotation specially (useful for
    some COVD training sets).

    .. note::

       This is a convenience function for our own dataset definitions inside
       this module, guaranteeting homogenity between dataset definitions
       provided in this package.  It assumes certain strategies for data
       augmentation that may not be translatable to other applications.


    Parameters
    ----------

    s : bob.ip.binseg.data.utils.SampleListDataset
        A dataset that will be augmented

    rotation_before : py:class:`bool`, Optional
        A optional flag allowing you to do a rotation augmentation transform
        **before** the sequence of transforms for this dataset, that will be
        augmented.


    Returns
    -------

    subset : :py:class:`bob.ip.binseg.data.utils.SampleListDataset`
        A pre-formatted dataset that can be fed to one of our engines

    """

    if rotation_before:
        return s.copy(RANDOM_ROTATION + s.transforms + RANDOM_FLIP_JITTER)

    return s.copy(s.transforms + RANDOM_ROTATION + RANDOM_FLIP_JITTER)


def make_dataset(subsets, transforms):
    """Creates a new configuration dataset from dictionary and transforms

    This function takes as input a dictionary as those that can be returned by
    :py:meth:`bob.ip.binseg.data.dataset.JSONDataset.subsets`,  or
    :py:meth:`bob.ip.binseg.data.dataset.CSVDataset.subsets`, mapping protocol
    names (such as ``train``, ``dev`` and ``test``) to
    :py:class:`bob.ip.binseg.data.sample.DelayedSample` lists, and a set of
    transforms, and returns a dictionary applying
    :py:class:`bob.ip.binseg.data.utils.SampleListDataset` to these
    lists, and our standard data augmentation if a ``train`` set exists.

    For example, if ``subsets`` is composed of two sets named ``train`` and
    ``test``, this function will yield a dictionary with the following entries:

    * ``__train__``: Wraps the ``train`` subset, includes data augmentation
      (note: datasets with names starting with ``_`` (underscore) are excluded
      from prediction and evaluation by default, as they contain data
      augmentation transformations.)
    * ``train``: Wraps the ``train`` subset, **without** data augmentation
    * ``train``: Wraps the ``test`` subset, **without** data augmentation

    .. note::

       This is a convenience function for our own dataset definitions inside
       this module, guaranteeting homogenity between dataset definitions
       provided in this package.  It assumes certain strategies for data
       augmentation that may not be translatable to other applications.


    Parameters
    ----------

    subsets : dict
        A dictionary that contains the delayed sample lists for a number of
        named lists.  If one of the keys is ``train``, our standard dataset
        augmentation transforms are appended to the definition of that subset.
        All other subsets remain un-augmented.  If one of the keys is
        ``validation``, then this dataset will be also copied to the
        ``__valid__`` hidden dataset and will be used for validation during
        training.  Otherwise, if no ``valid`` subset is available, we set
        ``__valid__`` to be the same as the unaugmented ``train`` subset, if
        one is available.

    transforms : list
        A list of transforms that needs to be applied to all samples in the set


    Returns
    -------

    dataset : dict
        A pre-formatted dataset that can be fed to one of our engines. It maps
        string names to
        :py:class:`bob.ip.binseg.data.utils.SampleListDataset`'s.

    """

    retval = {}

    for key in subsets.keys():
        retval[key] = make_subset(subsets[key], transforms=transforms)
        if key == "train":
            retval["__train__"] = make_subset(subsets[key],
                    transforms=transforms,
                    suffixes=(RANDOM_ROTATION + RANDOM_FLIP_JITTER),
                    )
        if key == "validation":
            # also use it for validation during training
            retval["__valid__"] = retval[key]

    if ("__train__" in retval) and ("train" in retval) \
            and ("__valid__" not in retval):
        # if the dataset does not have a validation set, we use the unaugmented
        # training set as validation set
        retval["__valid__"] = retval["train"]

    return retval
