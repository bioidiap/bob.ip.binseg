#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from .....common.data.transforms import Resize, ShrinkIntoSquare
    from ....data.drhagis import dataset as raw
    from .. import make_dataset as mk

    return mk(
        raw.subsets(protocol),
        [ShrinkIntoSquare(reference=2, threshold=0), Resize((1760, 1760))],
    )
