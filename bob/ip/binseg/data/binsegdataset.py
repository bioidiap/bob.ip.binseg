#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import random

class BinSegDataset(Dataset):
    """PyTorch dataset wrapper around bob.db binary segmentation datasets. 
    A transform object can be passed that will be applied to the image, ground truth and mask (if present). 
    It supports indexing such that dataset[i] can be used to get ith sample.
    
    Parameters
    ---------- 
    bobdb : :py:mod:`bob.db.base`
        Binary segmentation bob database (e.g. bob.db.drive) 
    split : str 
        ``'train'`` or ``'test'``. Defaults to ``'train'``
    transform : :py:mod:`bob.ip.binseg.data.transforms`, optional
        A transform or composition of transfroms. Defaults to ``None``.
    mask : bool
        whether dataset contains masks or not
    """
    def __init__(self, bobdb, split = 'train', transform = None,index_to = None):
        if index_to:
            self.database = bobdb.samples(split)[:index_to]
        else:
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
            dataitem [img_name, img, gt]
        """
        img = self.database[index].img.pil_image()
        gt = self.database[index].gt.pil_image()
        img_name = self.database[index].img.basename
        sample = [img, gt]
        
        if self.transform :
            sample = self.transform(*sample)
        
        sample.insert(0,img_name)
        
        return sample


class SSLBinSegDataset(Dataset):
    """PyTorch dataset wrapper around bob.db binary segmentation datasets. 
    A transform object can be passed that will be applied to the image, ground truth and mask (if present). 
    It supports indexing such that dataset[i] can be used to get ith sample.
    
    Parameters
    ---------- 
    labeled_dataset : :py:class:`torch.utils.data.Dataset`
        BinSegDataset with labeled samples
    unlabeled_dataset : :py:class:`torch.utils.data.Dataset`
        UnLabeledBinSegDataset with unlabeled data
    """
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
    

    def __len__(self):
        """
        Returns
        -------
        int
            size of the dataset
        """
        return len(self.labeled_dataset)
    
    def __getitem__(self,index):
        """
        Parameters
        ----------
        index : int
        
        Returns
        -------
        list
            dataitem [img_name, img, gt, unlabeled_img_name, unlabeled_img]
        """
        sample = self.labeled_dataset[index]
        unlabeled_img_name, unlabeled_img = self.unlabeled_dataset[0]
        sample.extend([unlabeled_img_name, unlabeled_img])
        return sample


class UnLabeledBinSegDataset(Dataset):
    # TODO: if switch to handle case were not a bob.db object but a path to a directory is used
    """PyTorch dataset wrapper around bob.db binary segmentation datasets. 
    A transform object can be passed that will be applied to the image, ground truth and mask (if present). 
    It supports indexing such that dataset[i] can be used to get ith sample.
    
    Parameters
    ---------- 
    dv : :py:mod:`bob.db.base` or str
        Binary segmentation bob database (e.g. bob.db.drive) or path to folder containing unlabeled images
    split : str 
        ``'train'`` or ``'test'``. Defaults to ``'train'``
    transform : :py:mod:`bob.ip.binseg.data.transforms`, optional
        A transform or composition of transfroms. Defaults to ``None``.
    """
    def __init__(self, db, split = 'train', transform = None,index_from= None):
        if index_from:
            self.database = db.samples(split)[index_from:]
        else:
            self.database = db.samples(split)
        self.transform = transform
        self.split = split   

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
            dataitem [img_name, img]
        """
        random.shuffle(self.database)
        img = self.database[index].img.pil_image()
        img_name = self.database[index].img.basename
        sample = [img]
        if self.transform :
            sample = self.transform(img)
        
        sample.insert(0,img_name)
        
        return sample