#!/usr/bin/env python
# coding=utf-8


"""Common utilities"""

import contextlib

import PIL.Image
import PIL.ImageOps
import PIL.ImageChops

import torch
import torch.utils.data

from .transforms import Compose, ToTensor


def invert_mode1_image(img):
    """Inverts a binary PIL image (mode == ``"1"``)"""

    return PIL.ImageOps.invert(img.convert("RGB")).convert(
        mode="1", dither=None
    )


def subtract_mode1_images(img1, img2):
    """Returns a new image that represents ``img1 - img2``"""

    return PIL.ImageChops.subtract(img1, img2)


def overlayed_image(
    img,
    label,
    mask=None,
    label_color=(0, 255, 0),
    mask_color=(0, 0, 255),
    alpha=0.4,
):
    """Creates an image showing existing labels and masko

    This function creates a new representation of the input image ``img``
    overlaying a green mask for labelled objects, and a red mask for parts of
    the image that should be ignored (negative mask).  By looking at this
    representation, it shall be possible to verify if the dataset/loader is
    yielding images correctly.


    Parameters
    ----------

    img : PIL.Image.Image
        An RGB PIL image that represents the original image for analysis

    label : PIL.Image.Image
        A PIL image in any mode that represents the labelled elements in the
        image.  In case of images in mode "L" or "1", white pixels represent
        the labelled object.  Black-er pixels represent background.

    mask : py:class:`PIL.Image.Image`, Optional
        A PIL image in mode "1" that represents the mask for the image.  White
        pixels indicate where content should be used, black pixels, content to
        be ignored.

    label_color : py:class:`tuple`, Optional
        A tuple with three integer entries indicating the RGB color to be used
        for labels.  Only used if ``label.mode`` is "1" or "L".

    mask_color : py:class:`tuple`, Optional
        A tuple with three integer entries indicating the RGB color to be used
        for the mask-negative (black parts in the original mask).

    alpha : py:class:`float`, Optional
        A float that indicates how much of blending should be performed between
        the label, mask and the original image.


    Returns
    -------

    image : PIL.Image.Image
        A new image overlaying the original image, object labels (in green) and
        what is to be considered parts to be **masked-out** (i.e. a
        representation of a negative of the mask).

    """

    # creates a representation of labels, in RGB format, with the right color
    if label.mode in ("1", "L"):
        label_colored = PIL.ImageOps.colorize(
            label.convert("L"), (0, 0, 0), label_color
        )
    else:
        # user has already passed an RGB version of the labels, just compose
        label_colored = label

    # blend image and label together - first blend to get vessels drawn with a
    # slight "label_color" tone on top, then composite with original image, to
    # avoid loosing brightness.
    retval = PIL.Image.blend(img, label_colored, alpha)
    if label.mode == "1":
        composite_mask = invert_mode1_image(label)
    else:
        composite_mask = PIL.ImageOps.invert(label.convert("L"))
    retval = PIL.Image.composite(img, retval, composite_mask)

    # creates a representation of the mask negative with the right color
    if mask is not None:
        antimask_colored = PIL.ImageOps.colorize(
            mask.convert("L"), mask_color, (0, 0, 0)
        )
        tmp = PIL.Image.blend(retval, antimask_colored, alpha)
        retval = PIL.Image.composite(retval, tmp, mask)

    return retval


class SampleListDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper around Sample lists

    A transform object can be passed that will be applied to the image, ground
    truth and mask (if present).

    It supports indexing such that dataset[i] can be used to get the i-th
    sample.


    Parameters
    ----------

    samples : list
        A list of :py:class:`bob.ip.binseg.data.sample.Sample` objects

    transforms : :py:class:`list`, Optional
        a list of transformations to be applied to **both** image and
        ground-truth data.  Notice a last transform
        (:py:class:`bob.ip.binseg.data.transforms.ToTensor`) is always applied
        - you do not need to add that.

    """

    def __init__(self, samples, transforms=[]):

        self._samples = samples
        self.transforms = transforms

    @property
    def transforms(self):
        return self._transforms.transforms[:-1]

    @transforms.setter
    def transforms(self, l):
        self._transforms = Compose(l + [ToTensor()])

    def copy(self, transforms=None):
        """Returns a deep copy of itself, optionally resetting transforms

        Parameters
        ----------

        transforms : :py:class:`list`, Optional
            An optional list of transforms to set in the copy.  If not
            specified, use ``self.transforms``.
        """

        return SampleListDataset(self._samples, transforms or self.transforms)

    def keys(self):
        """Generator producing all keys for all samples"""
        for k in self._samples:
            yield k.key

    def all_keys_match(self, other):
        """Compares all keys to ``other``, return ``True`` if all match
        """
        return len(self) == len(other) and all(
            [(ks == ko) for ks, ko in zip(self.keys(), other.keys())]
        )

    def __len__(self):
        """

        Returns
        -------

        size : int
            size of the dataset

        """
        return len(self._samples)

    def __getitem__(self, key):
        """

        Parameters
        ----------

        key : int, slice

        Returns
        -------

        sample : list
            The sample data: ``[key, image[, gt[, mask]]]``

        """

        if isinstance(key, slice):
            return [self[k] for k in range(*key.indices(len(self)))]
        else:  # we try it as an int
            item = self._samples[key]
            data = item.data  # triggers data loading

            retval = [data["data"]]
            if "label" in data:
                retval.append(data["label"])
            if "mask" in data:
                retval.append(data["mask"])

            if self._transforms:
                retval = self._transforms(*retval)

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
        self._labelled = labelled
        self._unlabelled = unlabelled

    def keys(self):
        """Generator producing all keys for all samples"""
        for k in self._labelled + self._unlabelled:
            yield k.key

    def all_keys_match(self, other):
        """Compares all keys to ``other``, return ``True`` if all match
        """
        return len(self) == len(other) and all(
            [(ks == ko) for ks, ko in zip(self.keys(), other.keys())]
        )

    def __len__(self):
        """

        Returns
        -------

        size : int
            size of the dataset

        """

        return len(self._labelled)

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

        retval = self._labelled[index]
        # gets one an unlabelled sample randomly to follow the labelled sample
        unlab = self._unlabelled[torch.randint(len(self._unlabelled), ())]
        # only interested in key and data
        return retval + unlab[:2]
