#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset

class BinSegDataset(Dataset):
    """PyTorch dataset wrapper around bob.db binary segmentation datasets. 
    A transform object can be passed that will be applied to the image, ground truth and mask (if present). 
    It supports indexing such that dataset[i] can be used to get ith sample.
    
    Attributes
    ---------- 
    bobdb : :py:mod:`bob.db.base`
        Binary segmentation bob database (e.g. bob.db.drive) 
    split : str 
        ``'train'`` or ``'test'``. Defaults to ``'train'``
    transform : :py:mod:`bob.ip.binseg.data.transforms`, optional
        A transform or composition of transfroms. Defaults to ``None``.
    """
    def __init__(self, bobdb, split = 'train', transform = None):
        self.database = bobdb.samples(split)
        self.transform = transform
        self.split = split
    
    @property
    def mask(self):
        # check if first sample contains a mask
        return hasattr(self.database[0], 'mask')

    def __len__(self):
        """
        Returns
        -------
        int
            size of the dataset
        """
        return len(self.database)
    
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
        img = self.database[index].img.pil_image()
        gt = self.database[index].gt.pil_image()
        img_name = self.database[index].img.basename
        sample = [img, gt]
        if self.mask:
            mask = self.database[index].mask.pil_image()
            sample.append(mask)
        
        if self.transform :
            sample = self.transform(*sample)
        
        sample.insert(0,img_name)
        
        return sample
