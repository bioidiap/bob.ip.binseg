#!/usr/bin/env python
# coding=utf-8

"""Dataset augmentation constants"""

from ...data.transforms import (
    RandomRotation,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
)

ROTATION = [RandomRotation()]
"""Shared data augmentation based on random rotation only"""

DEFAULT_WITHOUT_ROTATION = [
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    ColorJitter(),
]
"""Shared data augmentation transforms without random rotation"""

DEFAULT = ROTATION + DEFAULT_WITHOUT_ROTATION
"""Shared data augmentation transforms"""
