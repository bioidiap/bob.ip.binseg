#!/usr/bin/env python
# coding=utf-8


def _maker_1168(protocol):

    from .....common.data.transforms import Crop, Resize
    from ....data.hrf import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Crop(0, 108, 2336, 3296), Resize(1168)])


def _maker(protocol):

    from .....common.data.transforms import Crop
    from ....data.hrf import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Crop(0, 108, 2336, 3296)])


def _maker_square_768(protocol):

    from .....common.data.transforms import Pad, Resize
    from ....data.hrf import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 584)), Resize((768, 768))])


def _maker_square_1024(protocol):

    from .....common.data.transforms import Pad, Resize
    from ....data.hrf import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 584)), Resize((1024, 1024))])


def _maker_square(protocol, size):

    from .....common.data.transforms import Pad, Resize
    from ....data.hrf import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((0, 584)), Resize((size, size))])
