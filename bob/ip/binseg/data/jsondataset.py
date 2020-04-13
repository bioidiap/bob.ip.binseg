#!/usr/bin/env python
# coding=utf-8

import os
import copy
import json
import functools

import logging
logger = logging.getLogger(__name__)

from .sample import DelayedSample


class JSONDataset:
    """
    Generic multi-protocol filelist dataset that yields samples

    To create a new dataset, you need to provide one or more JSON formatted
    filelists (one per protocol) with the following contents:

    .. code-block:: json

       {
           "subset1": [
               {
                   "data": "path/to/data",
                   "label": "path/to/optional/label",
                   "mask": "path/to/optional/mask"
               }
           ],
           "subset2": [
           ]
       }

    Optionally, you may also format your JSON file like this, where each sample
    is described as a list of up to 3 elements:

    .. code-block:: json

       {
           "subset1": [
               [
                   "path/to/data",
                   "path/to/optional/label",
                   "path/to/optional/mask"
               ]
           ],
           "subset2": [
           ]
       }

    If your dataset does not have labels or masks, you may also represent it
    like this:

    .. code-block:: json

       {
           "subset1": [
               "path/to/data1",
               "path/to/data2"
           ],
           "subset2": [
           ]
       }

    Where:

    * ``data``: absolute or relative path leading to original image, in RGB
      format
    * ``label``: (optional) absolute or relative path with manual segmentation
      information.  This image will be converted to a binary image.  This
      dataset shall always yield label images in which white pixels (value=1)
      indicate the **presence** of the object, and black pixels (value=0), its
      absence.
    * ``mask``: (optional) absolute or relative path with a mask that indicates
      valid regions in the image where automatic segmentation should occur.
      This image will be converted to a binary image.  This dataset shall
      always yield mask images in which white pixels (value=1) indicate the
      **valid** regions of the mask, and black pixels (value=0), invalid parts.

    Relative paths are interpreted with respect to the location where the JSON
    file is or to an optional ``root_path`` parameter, that can be provided.

    There are no requirements concerning image or ground-truth homogenity.
    Anything that can be loaded by our image and data loaders is OK.

    Notice that all rows must have the same number of entries.

    To generate a dataset without ground-truth (e.g. for prediction tasks),
    then omit the ``label`` and ``mask`` entries.


    Parameters
    ----------

    protocols : [str]
        Paths to one or more JSON formatted files containing the various
        protocols to be recognized by this dataset.

    root_path : str
        Path to a common filesystem root where files with relative paths should
        be sitting.  If not set, then we use the current directory to resolve
        relative paths.

    loader : object
        A function that receives, as input, a context dictionary (with a
        "protocol" and "subset" keys indicating which protocol and subset are
        being served), and a dictionary with ``{key: path}`` entries, and
        returns a dictionary with the loaded data.  It shall respect the
        loading principles of data, label and mask objects as stated above.

    """

    def __init__(self, protocols, root_path, loader):

        self.protocols = dict(
            (os.path.splitext(os.path.basename(k))[0], os.path.realpath(k))
            for k in protocols
        )
        self.root_path = root_path
        self.loader = loader

    def check(self):
        """For each protocol, check all files are available on the filesystem

        Returns
        -------

        errors : int
            Number of errors found

        """

        logger.info(f"Checking dataset at '{self.root_path}'...")

        errors = 0
        for proto in self.protocols:
            logger.info(f"Checking protocol '{proto}'...")
            for name, samples in self.subsets(proto).items():
                logger.info(f"Checking subset '{name}'...")
                for sample in samples:
                    try:
                        sample.data  # triggers loading
                        logger.info(f"{sample.key}: OK")
                    except Exception as e:
                        logger.error(f"{sample.key}: {e}")
                        errors += 1
        return errors

    def subsets(self, protocol):
        """Returns all subsets in a protocol

        This method will load JSON information for a given protocol and return
        all subsets of the given protocol after converting each entry into a
        :py:class:`bob.ip.binseg.data.sample.DelayedSample`.

        Parameters
        ----------

        protocol : str
            Name of the protocol data to load


        Returns
        -------

        subsets : dict
            A dictionary mapping subset names to lists of
            :py:class:`bob.ip.binseg.data.sample.DelayedSample` objects, with
            the proper loading implemented.  Each delayed sample also carries a
            ``key`` parameter, that contains the relative path of the sample,
            without its extension.  This parameter can be used for recording
            sample transforms during check-pointing.

        """

        with open(self.protocols[protocol], "r") as f:
            data = json.load(f)

        # returns a fixed sample representations as a DelayedSamples
        retval = {}

        for subset, samples in data.items():
            delayeds = []
            context = dict(protocol=protocol, subset=subset)
            for k in samples:

                if isinstance(k, dict):
                    item = k

                elif isinstance(k, list):
                    item = {"data": k[0]}
                    if len(k) > 1: item["label"] = k[1]
                    if len(k) > 2: item["mask"] = k[2]

                elif isinstance(k, str):
                    item = {"data": k}

                key = os.path.splitext(item["data"])[0]

                # make paths absolute
                abs_item = copy.deepcopy(item)
                for k,v in item.items():
                    if not os.path.isabs(v):
                        abs_item[k] = os.path.join(self.root_path, v)

                load = functools.partial(self.loader, context, abs_item)
                delayeds.append(DelayedSample(load, key=key))

            retval[subset] = delayeds

        return retval
