#!/usr/bin/env python
# coding=utf-8


def _maker(protocol, raw=None):

    from ....data.stare import dataset as _raw
    from ....data.transforms import Pad

    raw = raw or _raw  # allows user to recreate dataset for testing purposes
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((2, 1, 2, 2))])


def _maker_square(protocol, raw=None):

    from ....data.stare import dataset as _raw
    from ....data.transforms import Pad, Resize

    raw = raw or _raw  # allows user to recreate dataset for testing purposes
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((1, 48, 0, 48)), Resize((768, 768))])
