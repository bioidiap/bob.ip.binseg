#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as VF
import bob.io.base

def get_file_lists(data_path, glob):
    """
    Recursively retrieves file lists from a given path, matching a given glob

    This function will use :py:meth:`pathlib.Path.rglob`, together with the
    provided glob pattern to search for anything the desired filename.
    """

    data_path = Path(data_path)
    image_file_names = np.array(sorted(list(data_path.rglob(glob))))
    return image_file_names

class ImageFolderInference(Dataset):
    """
    Generic ImageFolder containing images for inference

    Notice that this implementation, contrary to its sister
    :py:class:`.ImageFolder`, does not *automatically*
    convert the input image to RGB, before passing it to the transforms, so it
    is possible to accomodate a wider range of input types (e.g. 16-bit PNG
    images).

    Parameters
    ----------
    path : str
        full path to root of dataset

    glob : str
        glob that can be used to filter-down files to be loaded on the provided
        path

    transform : list
        List of transformations to apply to every input sample

    """
    def __init__(self, path, glob='*', transform = None):
        self.transform = transform
        self.path = path
        self.img_file_list = get_file_lists(path, glob)

    def __len__(self):
        """
        Returns
        -------
        int
            size of the dataset
        """
        return len(self.img_file_list)

    def __getitem__(self,index):
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        list
            dataitem [img_name, img]
        """
        img_path = self.img_file_list[index]
        img_name = img_path.relative_to(self.path).as_posix()
        img = Image.open(img_path)

        sample = [img]

        if self.transform :
            sample = self.transform(*sample)

        sample.insert(0,img_name)

        return sample
