#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import unittest
import numpy as np
from bob.ip.binseg.data.transforms import *

transforms = Compose([
                        RandomHFlip(prob=1)
                        ,RandomHFlip(prob=1)
                        ,RandomVFlip(prob=1)
                        ,RandomVFlip(prob=1)
                    ])

def create_img():
    t = torch.randn((3,42,24))
    pil = VF.to_pil_image(t)
    return pil


class Tester(unittest.TestCase):
    """
    Unit test for random flips
    """
    
    def test_flips(self):
        transforms = Compose([
                        RandomHFlip(prob=1)
                        ,RandomHFlip(prob=1)
                        ,RandomVFlip(prob=1)
                        ,RandomVFlip(prob=1)
                    ])
        img, gt, mask = [create_img() for i in range(3)]
        img_t, gt_t, mask_t = transforms(img, gt, mask)
        self.assertTrue(np.all(np.array(img_t) == np.array(img)))
        self.assertTrue(np.all(np.array(gt_t) == np.array(gt)))
        self.assertTrue(np.all(np.array(mask_t) == np.array(mask)))

    def test_to_tensor(self):
        transforms = ToTensor()
        img, gt, mask = [create_img() for i in range(3)]
        img_t, gt_t, mask_t = transforms(img, gt, mask)
        self.assertEqual(str(img_t.dtype),"torch.float32")
        self.assertEqual(str(gt_t.dtype),"torch.float32")
        self.assertEqual(str(mask_t.dtype),"torch.float32")

if __name__ == '__main__':
    unittest.main()