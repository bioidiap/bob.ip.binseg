#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.drionsdb import dataset as raw
    from ....data.transforms import Pad
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((4, 8, 4, 8))])
