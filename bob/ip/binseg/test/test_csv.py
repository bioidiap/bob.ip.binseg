#!/usr/bin/env python
# coding=utf-8

"""Unit tests for the CSV dataset"""

import io

import nose.tools

from ..data.dataset import CSVDataset
from ..data import stare

## special trick for CI builds
from . import mock_dataset, TESTDB_TMPDIR

datadir, json_dataset, rc_variable_set = mock_dataset()


## definition of stare subsets for "default" protocol
default = {
    "train": io.StringIO(
        """\
stare-images/im0001.ppm,labels-ah/im0001.ah.ppm
stare-images/im0002.ppm,labels-ah/im0002.ah.ppm
stare-images/im0003.ppm,labels-ah/im0003.ah.ppm
stare-images/im0004.ppm,labels-ah/im0004.ah.ppm
stare-images/im0005.ppm,labels-ah/im0005.ah.ppm
stare-images/im0044.ppm,labels-ah/im0044.ah.ppm
stare-images/im0077.ppm,labels-ah/im0077.ah.ppm
stare-images/im0081.ppm,labels-ah/im0081.ah.ppm
stare-images/im0082.ppm,labels-ah/im0082.ah.ppm
stare-images/im0139.ppm,labels-ah/im0139.ah.ppm"""
    ),
    "test": io.StringIO(
        """\
stare-images/im0162.ppm,labels-ah/im0162.ah.ppm
stare-images/im0163.ppm,labels-ah/im0163.ah.ppm
stare-images/im0235.ppm,labels-ah/im0235.ah.ppm
stare-images/im0236.ppm,labels-ah/im0236.ah.ppm
stare-images/im0239.ppm,labels-ah/im0239.ah.ppm
stare-images/im0240.ppm,labels-ah/im0240.ah.ppm
stare-images/im0255.ppm,labels-ah/im0255.ah.ppm
stare-images/im0291.ppm,labels-ah/im0291.ah.ppm
stare-images/im0319.ppm,labels-ah/im0319.ah.ppm
stare-images/im0324.ppm,labels-ah/im0324.ah.ppm"""
    ),
}


@rc_variable_set("bob.ip.binseg.stare.datadir")
def test_compare_to_json():

    test_dataset = CSVDataset(
        default,
        stare._fieldnames,
        stare._make_loader(datadir),
        stare.data_path_keymaker,
    )

    for subset in ("train", "test"):
        for t1, t2 in zip(
            test_dataset.samples(subset),
            json_dataset.subsets("ah")[subset],
        ):
            nose.tools.eq_(t1.key, t2.key)
            nose.tools.eq_(t1.data, t2.data)

    subsets = test_dataset.subsets()
    for subset in subsets.keys():
        for t1, t2 in zip(
            subsets[subset],
            json_dataset.subsets("ah")[subset],
        ):
            nose.tools.eq_(t1.key, t2.key)
            nose.tools.eq_(t1.data, t2.data)
