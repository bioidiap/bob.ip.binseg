#!/usr/bin/env python
# coding=utf-8

"""Dataset configuration utilities"""

from ...data.transforms import (
    RandomHFlip,
    RandomVFlip,
    RandomRotation,
    ColorJitter,
)

DATA_AUGMENTATION = [
        RandomHFlip(),
        RandomVFlip(),
        RandomRotation(),
        ColorJitter(),
        ]
"""Shared data augmentation transforms"""
