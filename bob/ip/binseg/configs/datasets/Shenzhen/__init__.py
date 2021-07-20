#!/usr/bin/env python
# coding=utf-8

def _maker(protocol):

    from ....data.transforms import Resize
    from ....data.Shenzhen import dataset as raw
    from .. import make_dataset as mk
    return mk(raw.subsets(protocol), [Resize((512, 512))])


def _maker_256(protocol):

    from ....data.transforms import Resize
    from ....data.Shenzhen import dataset as raw
    from .. import make_dataset as mk
    return mk(raw.subsets(protocol), [Resize((256, 256))])
