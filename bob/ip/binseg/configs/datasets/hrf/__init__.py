#!/usr/bin/env python
# coding=utf-8


def _maker_1168(protocol):

    from ....data.hrf import dataset as raw
    from ....data.transforms import Crop, Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Crop(0, 108, 2336, 3296), Resize(1168)])


def _maker(protocol):

    from ....data.hrf import dataset as raw
    from ....data.transforms import Crop
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Crop(0, 108, 2336, 3296)])
