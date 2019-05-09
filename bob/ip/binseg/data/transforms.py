#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision.transforms.functional as VF
import random
import PIL
from PIL import Image
from torchvision.transforms.transforms import Lambda
from torchvision.transforms.transforms import Compose as TorchVisionCompose
import math
from math import floor
import warnings


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

# Compose 

class Compose:
    """Composes several transforms.

    Attributes
    ----------
    transforms : list
        list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# Preprocessing

class CenterCrop:
    """
    Crop at the center.

    Attributes
    ----------
    size : int
        target size
    """
    def __init__(self, size):
        self.size = size
        
    def __call__(self, *args):
        return [VF.center_crop(img, self.size) for img in args]


class Crop:
    """
    Crop at the given coordinates.
    
    Attributes
    ----------
    i : int 
        upper pixel coordinate.
    j : int 
        left pixel coordinate.
    h : int 
        height of the cropped image.
    w : int 
        width of the cropped image.
    """
    def __init__(self, i, j, h, w):
        self.i = i
        self.j = j
        self.h = h 
        self.w = w 
    
    def __call__(self, *args):
        return [img.crop((self.j, self.i, self.j + self.w, self.i + self.h)) for img in args]

class Pad:
    """
    Constant padding

    Attributes
    ----------
    padding : int or tuple 
        padding on each border. If a single int is provided this is used to pad all borders. 
        If tuple of length 2 is provided this is the padding on left/right and top/bottom respectively.
        If a tuple of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
    
    fill : int
        pixel fill value for constant fill. Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively. 
        This value is only used when the padding_mode is constant   
    """
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill
        
    def __call__(self, *args):
        return [VF.pad(img, self.padding, self.fill, padding_mode='constant') for img in args]
    
class ToTensor:
    """Converts PIL.Image to torch.tensor """
    def __call__(self, *args):
        return [VF.to_tensor(img) for img in args]

        
# Augmentations

class RandomHFlip:
    """
    Flips horizontally
    
    Attributes
    ----------
    prob : float
        probability at which imgage is flipped. Defaults to ``0.5``
    """
    def __init__(self, prob = 0.5):
        self.prob = prob
        
    def __call__(self, *args):
        if random.random() < self.prob:
            return [VF.hflip(img) for img in args]
            
        else:
            return args
    
    
class RandomVFlip:
    """
    Flips vertically
    
    Attributes
    ----------
    prob : float 
        probability at which imgage is flipped. Defaults to ``0.5``
    """
    def __init__(self, prob = 0.5):
        self.prob = prob
        
    def __call__(self, *args):
        if random.random() < self.prob:
            return [VF.vflip(img) for img in args]
            
        else:
            return args
    

class RandomRotation:
    """
    Rotates by degree
    
    Attributes
    ----------
    degree_range : tuple
        range of degrees in which image and ground truth are rotated. Defaults to ``(-15, +15)``
    prob : float 
        probability at which imgage is rotated. Defaults to ``0.5``
    """
    def __init__(self, degree_range = (-15, +15), prob = 0.5):
        self.prob = prob
        self.degree_range = degree_range
        
    def __call__(self, *args):
        if random.random() < self.prob:
            degree = random.randint(*self.degree_range)
            return [VF.rotate(img, degree, resample = Image.BILINEAR) for img in args]
        else:
            return args

class ColorJitter(object):
    """ 
    Randomly change the brightness, contrast, saturation and hue
    
    Attributes
    ----------
    brightness : float 
        how much to jitter brightness. brightness_factor
        is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast : float
        how much to jitter contrast. contrast_factor
        is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation : float 
        how much to jitter saturation. saturation_factor
        is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue : float 
        how much to jitter hue. hue_factor is chosen uniformly from
        [-hue, hue]. Should be >=0 and <= 0.5
    prob : float
        probability at which the operation is applied
    """
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.02, hue=0.02, prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: VF.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: VF.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: VF.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: VF.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = TorchVisionCompose(transforms)

        return transform

    def __call__(self, *args):
        if random.random() < self.prob:
            transform = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
            trans_img = transform(args[0])
            return [trans_img, *args[1:]]
        else:
            return args


class RandomResizedCrop:
    """Crop to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    
    Attributes
    ----------
    size : int 
        expected output size of each edge
    scale : tuple 
        range of size of the origin size cropped. Defaults to ``(0.08, 1.0)``
    ratio : tuple
        range of aspect ratio of the origin aspect ratio cropped. Defaults to ``(3. / 4., 4. / 3.)``
    interpolation :
        Defaults to ``PIL.Image.BILINEAR``
    prob : float 
        probability at which the operation is applied. Defaults to ``0.5``
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, prob = 0.5):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.prob = prob

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = h * max(ratio)
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, *args):
        if random.random() < self.prob:
            imgs = []
            for img in args:
                i, j, h, w = self.get_params(img, self.scale, self.ratio)
                img = VF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
                imgs.append(img)
            return imgs
        else:
            return args

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class Distortion:
    """ 
    Applies random elastic distortion to a PIL Image, adapted from https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py : 
    As well as the probability, the granularity of the distortions
    produced by this class can be controlled using the width and
    height of the overlaying distortion grid. The larger the height
    and width of the grid, the smaller the distortions. This means
    that larger grid sizes can result in finer, less severe distortions.
    As well as this, the magnitude of the distortions vectors can
    also be adjusted.

    Attributes
    ----------
    grid_width : int 
        the width of the gird overlay, which is used by the class to apply the transformations to the image. Defaults to  ``8``
    grid_height : int 
        the height of the gird overlay, which is used by the class to apply the transformations to the image. Defaults to ``8``
    magnitude : int 
        controls the degree to which each distortion is applied to the overlaying distortion grid. Defaults to ``1``

    prob : float
        probability that the operation is performend. Defaults to ``0.5``
    """
    def __init__(self,grid_width=8, grid_height=8, magnitude=1, prob=0.5):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = magnitude
        self.prob = prob 
        
    def _generatemesh(self, image):
        w, h = image.size
        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height
        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))
        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))
        dimensions = []
        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles-1)+horizontal_tiles*i)
        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)
        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])
        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])
        for a, b, c, d in polygon_indices:
            dx = random.randint(-self.magnitude, self.magnitude)
            dy = random.randint(-self.magnitude, self.magnitude)
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]
        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        return generated_mesh
    
    def __call__(self,*args): 
        if random.random() < self.prob:
            # img, gt and mask have same resolution, we only generate mesh once:
            mesh = self._generatemesh(args[0])
            imgs = []
            for img in args:
                img = img.transform(img.size, Image.MESH, mesh, resample=Image.BICUBIC)
                imgs.append(img)
            return imgs
        else:
            return args