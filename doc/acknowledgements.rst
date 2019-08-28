.. -*- coding: utf-8 -*-
.. _bob.ip.binseg.acknowledgements:

================
Acknowledgements
================

This packages utilizes code from the following packages:

* The model-checkpointer is based on the Checkpointer in maskrcnn_benchmark by::

    @misc{massa2018mrcnn,
    author = {Massa, Francisco and Girshick, Ross},
    title = {{maskrcnn-benchmark: Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch}},
    year = {2018},
    howpublished = {\url{https://github.com/facebookresearch/maskrcnn-benchmark}},
    note = {Accessed: 2019.05.01}
    }

* The AdaBound optimizer code by::

    @inproceedings{Luo2019AdaBound,
     author = {Luo, Liangchen and Xiong, Yuanhao and Liu, Yan and Sun, Xu},
     title = {Adaptive Gradient Methods with Dynamic Bound of Learning Rate},
     booktitle = {Proceedings of the 7th International Conference on Learning Representations},
     month = {May},
     year = {2019},
     address = {New Orleans, Louisiana}
    }   

* The MobileNetV2 backbone is based on an implementation by::

    @misc{tonylins,
    author = {Ji Lin},
    title = {pytorch-mobilenet-v2},
    year = {2018}
    howpublished = {\url{https://github.com/tonylins/pytorch-mobilenet-v2}},
    note = {Accessed: 2019.05.01}
    }
