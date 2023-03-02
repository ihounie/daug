import itertools
import json
import logging
import math
import os
from collections import OrderedDict
import gc
import tempfile
import pickle
from dataclasses import dataclass
import random
from time import time

import numpy as np
import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import relu
from torch.utils.data import Subset, DataLoader

from tqdm import tqdm
import yaml
from theconf import Config as C, ConfigArgumentParser
from argparse import ArgumentParser

from TrivialAugment.common import get_logger
#from TrivialAugment.data_alb import get_dataloaders
from TrivialAugment.data import get_dataloaders
from TrivialAugment.lr_scheduler import adjust_learning_rate_resnet
from TrivialAugment.metrics import accuracy, Accumulator
from TrivialAugment.networks import get_model, num_class
#from TrivialAugment.train import run_epoch
from warmup_scheduler import GradualWarmupScheduler
import aug_lib
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
torch.backends.cudnn.benchmark = True
import wandb

logger = get_logger('TrivialAugment')
logger.setLevel(logging.DEBUG)

class ImageNetLightningModel(pl.LightningModule):
    def __init__(self, mh_steps=2, batch_size=64, model=None, criterion = nn.CrossEntropyLoss(reduction='none')):
        super().__init__()
        assert(model is not None)
        self.mh_steps = mh_steps
        self.batch_size = batch_size
        self.model = torch.jit.script(model)
        self.criterion = criterion
        self.dual_vars = 0
        self.margin = C.get()['PD']['margin']
        self.max_lr = C.get()['PD']['lr']
        self.dual_lr = self.max_lr/250
        print("*")
        print(f"dual lr {self.dual_lr}")

    def validation_step(self, batch, batch_idx):
        data, label = batch
        preds = self.model(data.to(memory_format=torch.channels_last))
        loss = self.criterion(preds, label)
        top1, top5 = accuracy(preds, label, (1, 5))
        self.log ("val/loss", loss.mean().item(), on_epoch=True)
        self.log ("val/top1", top1, on_epoch=True)
        self.log ("val/top5", top5, on_epoch=True)
        return {"loss": loss.mean().item(), "top1": top1, "top5" :top5}

    def on_train_epoch_end(self):
        self.dual_lr_scheduler_step()
        self.log("dual_lr", self.dual_lr)

    def dual_lr_scheduler_step(self):
        if self.current_epoch<3:
            self.dual_lr = self.max_lr/(250-self.current_epoch*50)
        elif self.current_epoch<100:
            self.dual_lr = self.max_lr/(100-self.current_epoch)
        elif self.current_epoch>180:
            self.dual_lr = 180*self.max_lr/(self.current_epoch)
        

    def training_step(self, batch, batch_idx):
        data, label = batch
        bs = label.shape[0]
        data = data
        #tinit = time()
        loss = self.criterion(self.model(torch.reshape(data,(-1, data.shape[2], data.shape[3], data.shape[4])).to(memory_format=torch.channels_last)), label.expand(data.shape[1], bs).flatten())
        clean_loss = torch.mean(loss[:bs])
        self.log("train/loss/clean", clean_loss, on_step=True, prog_bar=True)
        #print(f"Forward pass time: {time()-tinit}")
        ##########################################
        # Metropolis Hastings constraint sampling
        ##########################################
        #tinit = time()
        last_loss = loss[bs:2*bs]
        for _ in  range(self.mh_steps-1):
            proposal_loss = loss[(2+i)*bs:(3+i)*bs]
            acceptance_ratio = (
                torch.minimum((proposal_loss / last_loss), self.ones)
            )
            accepted = torch.bernoulli(acceptance_ratio).bool()
            last_loss[accepted] = proposal_loss[accepted]
        mh_loss = torch.mean(last_loss)
        self.log("train/loss/aug", mh_loss, on_step=True, prog_bar=True)
        #print(f"MH sampling: {time()-tinit}")
        # Primal descent
        loss =  clean_loss + self.dual_vars * mh_loss
        loss = loss/(1+self.dual_vars)
        ##############################
        #   Dual Ascent Step
        ##############################
        with torch.no_grad():
            self.dual_vars = relu(self.dual_vars + self.dual_lr * (mh_loss - self.margin))
        self.log("dual var", self.dual_vars, on_step=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        if C.get()['optimizer']['type'] == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=C.get()['lr'],
                momentum=C.get()['optimizer'].get('momentum', 0.9),
                weight_decay=C.get()['optimizer']['decay'],
                nesterov=C.get()['optimizer']['nesterov']
            )
        elif C.get()['optimizer']['type'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=C.get()['lr'],
                betas=(C.get()['optimizer'].get('momentum',.9),.999)
            )
        else:
            raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])

        lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
        if lr_scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epoch'], eta_min=0.)
        elif lr_scheduler_type == 'resnet':
            scheduler = adjust_learning_rate_resnet(optimizer)
        elif lr_scheduler_type == 'constant':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.)
        else:
            raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)
        if C.get()['lr_schedule'].get('warmup', None):
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
                total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
                after_scheduler=scheduler
            )
        return [optimizer], [scheduler]

class ImNetDataModule(pl.LightningDataModule):
    def __init__(self, train_dataloader,  val_dataloader):
        super().__init__()
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
    def train_dataloader(self):
        return self.train_dl
    def val_dataloader(self):
        return self.val_dl

def train_and_eval(dataroot,num_workers=1, test_ratio=0.0, cv_fold=0, logger=None):

    aug_lib.set_augmentation_space(C.get().get('augmentation_search_space', 'standard'), C.get().get('augmentation_parameter_max', 30), C.get().get('custom_search_space_augs', None))
    max_epoch = C.get()['epoch']
    trainsampler, trainloader, validloader, _, testtrainloader_, dataset_info = get_dataloaders("imagenet", C.get()['batch'], "~/imnet-data", num_workers=num_workers)
    datasets = ImNetDataModule(trainloader, validloader)
    # create a model & an optimizer
    model = get_model(C.get()['model'], C.get()['batch'], num_class(C.get()['dataset']))
    pl_model = ImageNetLightningModel(mh_steps=C.get()['MH']['steps'], batch_size= C.get()['batch'], model=model.to(memory_format=torch.channels_last))
    trainer = pl.Trainer(gpus=1, precision=16, accumulate_grad_batches=int(2048/(C.get()['batch'])), logger=logger)
    trainer.validate(pl_model, datasets)
    trainer.fit(pl_model, datasets)
    return


def parse_args():
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='Constrained')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels',
                        help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('-nw', '--num_workers', type=int, default=10)
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('-c','--config', type=str, default="confs/imnet/resnet50_imagenet_270epochs_4x8xb64_fixedlr_ta_fixed.yaml")
    parser.add_argument('--project', type=str, default='Daug-Imnet', help='wandb project')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.config is not None:
        try:
            C(args.config[0])
            print("conf successfully loaded")
        except:
            print("conf error")
    if 'seed' in C.get():
        seed = C.get()['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        wandb_logger = WandbLogger(project=args.project, name=args.tag)
        wandb.config.update(args)
        wandb.config.update(C.get().flatten())
        train_and_eval(args.dataroot,num_workers=args.num_workers, test_ratio=0.0, cv_fold=0, logger=wandb_logger)
    