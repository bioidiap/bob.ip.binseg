#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as VF
import bob.io.base

def get_file_lists(data_path):
    data_path = Path(data_path)
    
    image_path = data_path.joinpath('images')
    image_file_names = np.array(sorted(list(image_path.glob('*'))))

    gt_path = data_path.joinpath('gt')
    gt_file_names = np.array(sorted(list(gt_path.glob('*'))))
    return image_file_names, gt_file_names

class ImageFolder(Dataset):
    """
    Generic ImageFolder dataset, that contains two folders:

    * ``images`` (vessel images)
    * ``gt`` (ground-truth labels)
    
    
    Parameters
    ----------
    path : str
        full path to root of dataset
    
    """
    def __init__(self, path, transform = None):
        self.transform = transform
        self.img_file_list, self.gt_file_list = get_file_lists(path)

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
            dataitem [img_name, img, gt, mask]
        """
        img_path = self.img_file_list[index]
        img_name = img_path.name
        img = Image.open(img_path).convert(mode='RGB')
    
        gt_path = self.gt_file_list[index]
        if gt_path.suffix == '.hdf5':
            gt = bob.io.base.load(str(gt_path)).astype('float32')
            # not elegant but since transforms require PIL images we do this hacky workaround here
            gt = torch.from_numpy(gt)
            gt = VF.to_pil_image(gt).convert(mode='1', dither=None)
        else:
            gt = Image.open(gt_path).convert(mode='1', dither=None)
        
        sample = [img, gt]
        
        if self.transform :
            sample = self.transform(*sample)
        
        sample.insert(0,img_name)
        
        return sample
