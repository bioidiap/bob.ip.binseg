#!/usr/bin/env python
# coding=utf-8

import os
import csv
import copy
import json
import pathlib
import functools

import logging

logger = logging.getLogger(__name__)

from .sample import DelayedSample


class JSONDataset:
    """
    Generic multi-protocol/subset filelist dataset that yields samples

    To create a new dataset, you need to provide one or more JSON formatted
    filelists (one per protocol) with the following contents:

    .. code-block:: json

       {
           "subset1": [
               [
                   "value1",
                   "value2",
                   "value3"
               ],
               [
                   "value4",
                   "value5",
                   "value6"
               ]
           ],
           "subset2": [
           ]
       }

    Your dataset many contain any number of subsets, but all sample entries
    must contain the same number of fields.


    Parameters
    ----------

    protocols : list, dict
        Paths to one or more JSON formatted files containing the various
        protocols to be recognized by this dataset, or a dictionary, mapping
        protocol names to paths (or opened file objects) of CSV files.
        Internally, we save a dictionary where keys default to the basename of
        paths (list input).

    fieldnames : list, tuple
        An iterable over the field names (strings) to assign to each entry in
        the JSON file.  It should have as many items as fields in each entry of
        the JSON file.

    loader : object
        A function that receives as input, a context dictionary (with at least
        a "protocol" and "subset" keys indicating which protocol and subset are
        being served), and a dictionary with ``{key: path}`` entries, and
        returns a dictionary with the loaded data.

    keymaker : object
        A function that receives as input the same input from the ``loader``,
        but outputs a single string that uniquely identifies a sample within
        a given protocol.  It is typically the path, without extension, of one
        of the file entries for the sample, but you can tune it as you like.

    """

    def __init__(self, protocols, fieldnames, loader, keymaker):

        if isinstance(protocols, dict):
            self.protocols = protocols
        else:
            self.protocols = dict(
                (os.path.splitext(os.path.basename(k))[0], k)
                for k in protocols
            )
        self.fieldnames = fieldnames
        self.loader = loader
        self.keymaker = keymaker

    def check(self):
        """For each protocol, check if all data can be correctly accessed

        Returns
        -------

        errors : int
            Number of errors found

        """

        logger.info(f"Checking dataset...")
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

    def _make_delayed(self, pos, sample, context):
        """Checks consistence and builds a delayed loading sample
        """
        assert len(sample) == len(self.fieldnames), (
            f"Entry {k} in subset {context['subset']} of protocol "
            f"{context['protocol']} has {len(sample)} entries instead of "
            f"{len(self.fieldnames)} (expected). Fix file "
            f"{self.protocols[context['protocol']]}"
        )
        item = dict(zip(self.fieldnames, sample))
        return DelayedSample(
            functools.partial(self.loader, context, item),
            key=self.keymaker(context, item),
        )

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
            ``key`` parameter, that contains the output of the sample
            contextual data after passing through the ``keymaker``.  This
            parameter can be used for recording sample transforms during
            check-pointing.

        """

        fileobj = self.protocols[protocol]
        if isinstance(fileobj, (str, bytes, pathlib.Path)):
            with open(self.protocols[protocol], "r") as f:
                data = json.load(f)
        else:
            data = json.load(f)
            fileobj.seek(0)

        retval = {}
        for subset, samples in data.items():
            context = dict(protocol=protocol, subset=subset)
            retval[subset] = [
                self._make_delayed(k, v, context) for (k, v) in enumerate(samples)
            ]
        return retval


class CSVDataset:
    """
    Generic single subset filelist dataset that yields samples

    To create a new dataset, you only need to provide a CSV formatted filelist
    using any separator (e.g. comma, space, semi-colon) with the following
    information:

    .. code-block:: text

       value1,value2,value3
       value4,value5,value6
       ...

    Notice that all rows must have the same number of entries.

    Parameters
    ----------

    subsets : list, dict
        Paths to one or more CSV formatted files containing the various subsets
        to be recognized by this dataset, or a dictionary, mapping subset names
        to paths (or opened file objects) of CSV files.  Internally, we save a
        dictionary where keys default to the basename of paths (list input).

    fieldnames : list, tuple
        An iterable over the field names (strings) to assign to each column in
        the CSV file.  It should have as many items as fields in each row of
        the CSV file(s).

    loader : object
        A function that receives as input, a context dictionary (with, at
        least, a "subset" key indicating which subset is being served), and a
        dictionary with ``{key: path}`` entries, and returns a dictionary with
        the loaded data.

    keymaker : object
        A function that receives as input the same input from the ``loader``,
        but outputs a single string that uniquely identifies a sample within
        a given protocol.  It is typically the path, without extension, of one
        of the file entries for the sample, but you can tune it as you like.

    """

    def __init__(self, subsets, fieldnames, loader, keymaker):

        if isinstance(subsets, dict):
            self._subsets = subsets
        else:
            self._subsets = dict(
                (os.path.splitext(os.path.basename(k))[0], k)
                for k in subsets
            )
        self.fieldnames = fieldnames
        self.loader = loader
        self.keymaker = keymaker

    def check(self):
        """For each subset, check if all data can be correctly accessed

        Returns
        -------

        errors : int
            Number of errors found

        """

        logger.info(f"Checking dataset...")
        errors = 0
        for name in self._subsets.keys():
            logger.info(f"Checking subset '{name}'...")
            for sample in self.samples(name):
                try:
                    sample.data  # triggers loading
                    logger.info(f"{sample.key}: OK")
                except Exception as e:
                    logger.error(f"{sample.key}: {e}")
                    errors += 1
        return errors

    def _make_delayed(self, pos, sample, context):
        """Checks consistence and builds a delayed loading sample
        """
        assert len(sample) == len(self.fieldnames), (
            f"Entry {k} in subset {context['subset']} has {len(sample)} "
            f"entries instead of {len(self.fieldnames)} (expected). Fix "
            f"file {self._subsets[context['subset']]}"
        )
        item = dict(zip(self.fieldnames, sample))
        return DelayedSample(
            functools.partial(self.loader, context, item),
            key=self.keymaker(context, item),
        )

    def subsets(self):
        """Returns all available subsets at once

        Returns
        -------

        subsets : dict
            A dictionary mapping subset names to lists of
            :py:class:`bob.ip.binseg.data.sample.DelayedSample` objects, with
            the proper loading implemented.  Each delayed sample also carries a
            ``key`` parameter, that contains the output of the sample
            contextual data after passing through the ``keymaker``.  This
            parameter can be used for recording sample transforms during
            check-pointing.

        """

        return dict((k, self.samples(k)) for k in self._subsets.keys())

    def samples(self, subset):
        """Returns all samples in a subset

        This method will load CSV information for a given subset and return
        all samples of the given subset after converting each entry into a
        :py:class:`bob.ip.binseg.data.sample.DelayedSample`.


        Parameters
        ----------

        subset : str
            Name of the subset data to load


        Returns
        -------

        subset : list
            A list of :py:class:`bob.ip.binseg.data.sample.DelayedSample`
            objects, with the proper loading implemented.  Each delayed sample
            also carries a ``key`` parameter, that contains the output of the
            sample contextual data after passing through the ``keymaker``.
            This parameter can be used for recording sample transforms during
            check-pointing.

        """

        fileobj = self._subsets[subset]
        if isinstance(fileobj, (str, bytes, pathlib.Path)):
            with open(self._subsets[subset], newline="") as f:
                cf = csv.reader(f)
                samples = [k for k in cf]
        else:
            cf = csv.reader(fileobj)
            samples = [k for k in cf]
            fileobj.seek(0)

        context = dict(subset=subset)
        return [self._make_delayed(k, v, context) for (k, v) in enumerate(samples)]
