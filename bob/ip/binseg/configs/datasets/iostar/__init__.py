#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.iostar import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [])


def _maker_square(protocol, size):

    from .....common.data.transforms import Resize
    from ....data.iostar import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Resize((size, size))])
