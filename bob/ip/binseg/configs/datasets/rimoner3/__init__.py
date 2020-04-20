#!/usr/bin/env python
# coding=utf-8

def _maker(protocol):

    from ....data.transforms import Pad
    from ....data.rimoner3 import dataset as raw
    from .. import make_dataset as mk
    return mk(raw.subsets(protocol), [Pad((8, 8, 8, 8))])
