#!/usr/bin/env python
# coding=utf-8


def _maker(protocol):

    from ....data.jsrt import dataset as raw
    from ....data.transforms import Resize
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Resize((512, 512))])


def _maker_augmented(protocol):

    from ....data.jsrt import dataset as raw
    from ....data.transforms import Compose as _compose
    from ....data.transforms import Resize as _resize
    from ....data.transforms import ColorJitter as _jitter
    from ....data.transforms import RandomHorizontalFlip as _hflip
    from ....data.transforms import RandomRotation as _rotation
    from ....data.transforms import GaussianBlur as _blur

    from .. import make_augmented_dataset as mad

    return mad(subsets=raw.subsets(protocol),
               all_transforms=[_resize((256, 256))],
               train_transforms=[_compose([
                                          _resize((256, 256)),
                                          _rotation(degrees=15, p=0.5),
                                          _hflip(p=0.5),
                                          _jitter(p=0.5),
                                          _blur(p=0.5),])])
