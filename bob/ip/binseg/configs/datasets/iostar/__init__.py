#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.iostar import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [])


def _maker_vessel_square(protocol):

    from ....data.iostar import dataset as raw
    from ....data.transforms import Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Resize((768, 768))])


def _maker_od_square(protocol):

    from ....data.iostar import dataset as raw
    from ....data.transforms import Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Resize((512, 512))])
