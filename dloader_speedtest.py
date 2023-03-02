from TrivialAugment.data import get_dataloaders
from theconf import Config as C, ConfigArgumentParser
from time import time
from argparse import ArgumentParser
import torch

parser= ArgumentParser()
parser.add_argument('-c',"--config_path", default="confs/imnet/resnet50_imagenet_270epochs_4x8xb64_fixedlr_ta_fixed.yaml", type=str, help="config path")
parser.add_argument('-b',"--batch_size", default=64, type=int)
parser.add_argument('-n',"--num_batches", default=50, type=int)
parser.add_argument('-wup',"--warmup", default=5, type=int)
parser.add_argument('-n_aug',"--num_aug", default=2, type=int)
args = parser.parse_args()
try:
    C(args.config_path)
    print("conf successfully loaded")
except:
    print("theconf singleton error - ignore it")
for workers in range( 30, 0, -1):
    trainsampler, trainloader, validloader, testloader_, testtrainloader_, dataset_info = get_dataloaders("imagenet", args.batch_size, "/home/chiche/imnet/imagenet1k/ILSVRC/Data/CLS-LOC", num_workers=workers)
    start = time()
    i = 0
    for batch in trainloader: # logging loader might be a loader or a loader wrapped into tqdm
        data, label = batch[0].cuda().to(memory_format=torch.channels_last), batch[1].cuda()
        if i>args.num_batches+args.warmup:
            break
        elif i==args.warmup:
            start = time()    
        i+=1
    time_delta = time()-start
    print(f" Num workers: {workers}, time: {time_delta}")
    epoch_time =time_delta*1281167/args.batch_size/args.num_batches/60/60
    print(f"Epoch time: {epoch_time}")
    print(f"Training Days: {270*epoch_time/24}")
