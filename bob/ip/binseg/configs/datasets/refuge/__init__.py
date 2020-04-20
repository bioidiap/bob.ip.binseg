#!/usr/bin/env python
# coding=utf-8

def _maker(protocol):

    from ....data.transforms import Pad, Resize, CenterCrop
    from ....data.refuge import dataset as raw
    from .. import make_dataset as mk
    # due to different sizes, we need to make the dataset twice
    train = mk(raw.subsets(protocol), [Resize(1539), Pad((21, 46, 22, 47))])
    # we'll keep "dev" and "test" from the next one
    retval = mk(raw.subsets(protocol), [CenterCrop(1632)])
    # and we keep the "train" set with the right transforms
    retval["train"] = train["train"]
    return retval
