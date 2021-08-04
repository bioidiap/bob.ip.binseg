#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.chasedb1 import dataset as raw
    from ....data.transforms import Crop
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Crop(0, 18, 960, 960)])
