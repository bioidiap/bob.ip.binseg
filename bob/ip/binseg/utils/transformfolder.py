#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path,PurePosixPath
from PIL import Image
from torchvision.transforms.functional import to_pil_image

def transformfolder(source_path, target_path, transforms):
    """Applies a set of transfroms on an image folder 
    
    Parameters
    ----------
    source_path : str
        [description]
    target_path : str
        [description]
    transforms : [type]
        transform function
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    file_paths = sorted(list(source_path.glob('*?.*')))
    for f in file_paths:
        timg_path = PurePosixPath(target_path).joinpath(f.name)
        img = Image.open(f).convert(mode='1', dither=None)
        img, _ = transforms(img,img)
        img = to_pil_image(img)
        img.save(str(timg_path))