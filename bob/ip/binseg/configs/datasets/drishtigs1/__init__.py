#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.drishtigs1 import dataset as raw
    from ....data.transforms import CenterCrop as ccrop
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [ccrop((1760, 2048))])


def _maker_square_512(protocol):

    from ....data.drishtigs1 import dataset as raw
    from ....data.transforms import CenterCrop as ccrop
    from ....data.transforms import Pad, Resize
    from .. import make_dataset as mk

    return mk(
        raw.subsets(protocol),
        [ccrop((1760, 2048)), Pad((0, 144)), Resize((512, 512))],
    )


def _maker_square_768(protocol):

    from ....data.drishtigs1 import dataset as raw
    from ....data.transforms import CenterCrop as ccrop
    from ....data.transforms import Pad, Resize
    from .. import make_dataset as mk

    return mk(
        raw.subsets(protocol),
        [ccrop((1760, 2048)), Pad((0, 144)), Resize((768, 768))],
    )
