#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.drive import Database as DRIVE
from bob.ip.binseg.data.binsegdataset import BinSegDataset
from bob.ip.binseg.data.transforms import ToTensor
from bob.ip.binseg.engine.trainer import do_train
from bob.ip.binseg.modeling.driu import build_driu
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from bob.ip.binseg.utils.checkpointer import Checkpointer, DetectronCheckpointer
from torch.nn import BCEWithLogitsLoss

import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("bob.ip.binseg.engine.train")

def train():
    # bob.db.dataset init
    drive = DRIVE()
    
    # Transforms 
    transforms = ToTensor()

    # PyTorch dataset
    bsdataset = BinSegDataset(drive,split='train', transform=transforms)
    
    # Build model
    model = build_driu()
    
    # Dataloader
    data_loader = DataLoader(
        dataset = bsdataset
        ,batch_size = 2
        ,shuffle= True
        ,pin_memory = False
        )
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    # criterion
    criterion = BCEWithLogitsLoss()

    # scheduler
    scheduler = MultiStepLR(optimizer, milestones=[150], gamma=0.1)

    # checkpointer
    checkpointer = DetectronCheckpointer(model, optimizer, scheduler,save_dir = "./output_temp", save_to_disk=True)

    # checkpoint period
    checkpoint_period = 2

    # pretrained backbone
    pretraind_backbone = model_urls['vgg16']

    # device 
    device = "cpu"
    
    # arguments 
    arguments = {}
    arguments["epoch"] = 0 
    arguments["max_epoch"] = 6
    extra_checkpoint_data = checkpointer.load(pretraind_backbone)
    arguments.update(extra_checkpoint_data)
    logger.info("Training for {} epochs".format(arguments["max_epoch"]))
    logger.info("Continuing from epoch {}".format(arguments["epoch"]))
    do_train(model
            , data_loader
            , optimizer
            , criterion
            , scheduler
            , checkpointer
            , checkpoint_period
            , device
            , arguments)


if __name__ == '__main__':
    train()