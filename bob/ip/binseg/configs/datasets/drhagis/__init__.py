#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.drhagis import dataset as raw
    from ....data.transforms import Resize, ResizeCrop
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [ResizeCrop(), Resize((1760, 1760))])
