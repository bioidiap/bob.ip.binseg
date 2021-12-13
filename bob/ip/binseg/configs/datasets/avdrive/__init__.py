#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.drive import dataset as raw
    from ....data.transforms import CenterCrop as ccrop
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [ccrop((544, 544))])
