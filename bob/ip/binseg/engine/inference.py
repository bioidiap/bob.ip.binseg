#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.drive import Database as DRIVE
from bob.ip.binseg.data.binsegdataset import BinSegDataset
from bob.ip.binseg.data.transforms import ToTensor
from bob.ip.binseg.engine.inferencer import do_inference
from bob.ip.binseg.modeling.driu import build_driu
from torch.utils.data import DataLoader
from bob.ip.binseg.utils.checkpointer import Checkpointer


import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("bob.ip.binseg.engine.inferencer")

def inference():
    # bob.db.dataset init
    drive = DRIVE()
    
    # Transforms 
    transforms = ToTensor()

    # PyTorch dataset
    bsdataset = BinSegDataset(drive,split='test', transform=transforms)
    
    # Build model
    model = build_driu()
    
    # Dataloader
    data_loader = DataLoader(
        dataset = bsdataset
        ,batch_size = 2
        ,shuffle= False
        ,pin_memory = False
        )
    
    # checkpointer, load last model in dir
    checkpointer = Checkpointer(model, save_dir = "./output_temp", save_to_disk=False)
    checkpointer.load()

    # device 
    device = "cpu"
    logger.info("Run inference and calculate evaluation metrics")
    do_inference(model
            , data_loader
            , device
            , "./output_temp"
            )


if __name__ == '__main__':
    inference()