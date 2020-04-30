#!/usr/bin/env python
# coding=utf-8

def _maker(protocol):

    from ....data.transforms import CenterCrop as ccrop
    from ....data.drishtigs1 import dataset as raw
    from .. import make_dataset as mk
    return mk(raw.subsets(protocol), [ccrop((1760, 2048))])
