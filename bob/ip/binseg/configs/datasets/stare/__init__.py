#!/usr/bin/env python
# coding=utf-8


def _maker(protocol, raw=None):

    from .....common.data.transforms import Pad
    from ....data.stare import dataset as _raw

    raw = raw or _raw  # allows user to recreate dataset for testing purposes
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((2, 1, 2, 2))])


def _maker_square(protocol, size, raw=None):

    from .....common.data.transforms import Pad, Resize
    from ....data.stare import dataset as _raw

    raw = raw or _raw  # allows user to recreate dataset for testing purposes
    from .. import make_dataset as mk

    return mk(
        raw.subsets(protocol), [Pad((1, 48, 0, 48)), Resize((size, size))]
    )
