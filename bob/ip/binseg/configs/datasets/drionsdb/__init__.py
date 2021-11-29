#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.drionsdb import dataset as raw
    from ....data.transforms import Pad
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((4, 8, 4, 8))])


def _maker_square_512(protocol):

    from ....data.drionsdb import dataset as raw
    from ....data.transforms import Pad, Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 100)), Resize((512, 512))])


def _maker_square_768(protocol):

    from ....data.drionsdb import dataset as raw
    from ....data.transforms import Pad, Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 100)), Resize((768, 768))])
