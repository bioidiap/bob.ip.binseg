#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset

class BinSegDataset(Dataset):
    """
    PyTorch dataset wrapper around bob.db binary segmentation datasets. 
    A transform object can be passed that will be applied to the image, ground truth and mask (if present). 
    
    It supports indexing such that dataset[i] can be used to get ith sample, e.g.: 
    img, gt, mask, name = db[0]
    
    Parameters
    ----------
    database  : binary segmentation `bob.db.database`
               
    split     : str
                    train' or 'test'

    transform : :py:class:`bob.ip.binseg.data.transforms.Compose`
 
    """
    def __init__(self, bobdb, split = None, transform = None):
        self.database = bobdb.samples(split)
        self.transform = transform
        self.split = split
    
    def __len__(self):
        """
        Returns the size of the dataset
        """
        return len(self.database)
    
    def __getitem__(self,index):
        img = self.database[index].img.pil_image()
        gt = self.database[index].gt.pil_image()
        mask = self.database[index].mask.pil_image() if hasattr(self.database[index], 'mask') else None
        img_name = self.database[index].img.basename
        
        if self.transform and mask:
            img, gt, mask = self.transform(img, gt, mask)
        else:
            img, gt  = self.transform(img, gt)
            
        return img, gt, mask, img_name
