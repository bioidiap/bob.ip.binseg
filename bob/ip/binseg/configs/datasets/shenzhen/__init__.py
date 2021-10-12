#!/usr/bin/env python
# coding=utf-8


def _maker(protocol, n):

    from ....data.shenzhen import dataset as raw
    from ....data.transforms import Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Resize((n, n))])
