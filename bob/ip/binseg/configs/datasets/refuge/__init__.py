#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.refuge import dataset as raw
    from ....data.transforms import CenterCrop, Pad, Resize
    from .. import make_dataset as mk

    # due to different sizes, we need to make the dataset twice
    train = mk(raw.subsets(protocol), [Resize(1539), Pad((21, 46, 22, 47))])
    # we'll keep "dev" and "test" from the next one
    retval = mk(raw.subsets(protocol), [CenterCrop(1632)])
    # and we keep the "train" set with the right transforms
    retval["train"] = train["train"]
    return retval


def _maker_square_512(protocol):

    from ....data.refuge import dataset as raw
    from ....data.transforms import CenterCrop, Pad, Resize
    from .. import make_dataset as mk

    # due to different sizes, we need to make the dataset twice
    train = mk(
        raw.subsets(protocol),
        [Resize(1539), Pad((21, 46, 22, 47)), Resize((512, 512))],
    )
    # we'll keep "dev" and "test" from the next one
    retval = mk(raw.subsets(protocol), [CenterCrop(1632), Resize((512, 512))])
    # and we keep the "train" set with the right transforms
    retval["train"] = train["train"]
    return retval


def _maker_square_768(protocol):

    from ....data.refuge import dataset as raw
    from ....data.transforms import CenterCrop, Pad, Resize
    from .. import make_dataset as mk

    # due to different sizes, we need to make the dataset twice
    train = mk(
        raw.subsets(protocol),
        [Resize(1539), Pad((21, 46, 22, 47)), Resize((768, 768))],
    )
    # we'll keep "dev" and "test" from the next one
    retval = mk(raw.subsets(protocol), [CenterCrop(1632), Resize((768, 768))])
    # and we keep the "train" set with the right transforms
    retval["train"] = train["train"]
    return retval
