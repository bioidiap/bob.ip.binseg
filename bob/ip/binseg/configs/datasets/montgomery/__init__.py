#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.montgomery import dataset as raw
    from ....data.transforms import Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Resize((512, 512))])
