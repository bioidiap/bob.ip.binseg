#!/usr/bin/env python
# coding=utf-8

def _maker(protocol):

    from ....data.transforms import Resize
    from ....data.drhagis import dataset as raw
    from .. import make_dataset as mk
    return mk(raw.subsets(protocol), [Resize((1760,1760))])

