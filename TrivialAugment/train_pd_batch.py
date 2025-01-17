import itertools
import json
import logging
import math
import os
from pathlib import Path
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
from torch.nn.functional import relu, cross_entropy
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import yaml
from theconf import Config as C, ConfigArgumentParser
from argparse import ArgumentParser
import pandas as pd
import wandb

from TrivialAugment.common import get_logger
from TrivialAugment.data import get_dataloaders
from TrivialAugment.lr_scheduler import adjust_learning_rate_resnet
from TrivialAugment.metrics import accuracy, Accumulator
from TrivialAugment.networks import get_model, num_class
from TrivialAugment.train import run_epoch
from warmup_scheduler import GradualWarmupScheduler
import aug_lib
from aug_lib import AugMeter

logger = get_logger('TrivialAugment')
logger.setLevel(logging.DEBUG)

def run_epoch_dual(rank, worldsize, model, loader, loss_fn, optimizer, dual_vars, wandb_log = True, augmented_dset=None, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None,sample_pairing_loader=None, log_interval=100):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))    # KakaoBrain Environment
    if verbose:
        logging_loader = tqdm(loader, disable=tqdm_disable)
        logging_loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))
    else:
        logging_loader = loader

    metrics = Accumulator()
    cnt = 0
    eval_cnt = 0
    total_steps = len(loader)
    steps = 0
    sample_constraint = C.get()['PD'].get('sample')
    n_aug = C.get()['n_aug']

    gc.collect()
    torch.cuda.empty_cache()
    #print('mem usage', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    communicate_grad_every = C.get().get('communicate_grad_every', 1)
    before_load_time = time()
    if C.get().get('load_sample_pairing_batch',False) and sample_pairing_loader is not None:
        sample_pairing_iter = iter(sample_pairing_loader)
        aug_lib.blend_images = [transforms.ToPILImage()(sample_pairing_loader.denorm(ti)) for ti in
                                next(sample_pairing_iter)[0]]
    aug_stats =  AugMeter()
    for batch in logging_loader: # logging loader might be a loader or a loader wrapped into tqdm
        data, label, indexes = batch[:3]
        steps += 1
        if C.get().get('load_sample_pairing_batch',False) and sample_pairing_loader is not None:
            try:
                aug_lib.blend_images = [transforms.ToPILImage()(sample_pairing_loader.denorm(ti)) for ti in next(sample_pairing_iter)[0]]
            except StopIteration:
                print("Blend images iterator ended. If this is printed twice per loop, there is something out-of-order.")
                pass
        data, label = data.to(f"cuda:{rank}"), label.to(f"cuda:{rank}")


        communicate_grad = steps % communicate_grad_every == 0
        just_communicated_grad = steps % communicate_grad_every == 1 # also is true in first step of each epoch
        if optimizer and (communicate_grad_every == 1 or just_communicated_grad):
            optimizer.zero_grad()

        batch_subset = Subset(augmented_dset, indexes)
        aug_loader = DataLoader(batch_subset, batch_size=indexes.shape[0], shuffle=False, num_workers=0, pin_memory=True,drop_last=False)

        ##########################################
        # Metropolis Hastings constraint sampling
        ##########################################
        mh_steps = C.get()['MH'].get('steps')
        # First step
        aug_data, _, op, level = next(iter(aug_loader))
        if worldsize > 1:
                aug_data = aug_data.to(rank)
        else:
            aug_data = aug_data.to(f"cuda:{rank}")
        batch_size = aug_data.shape[0]
        proposals = torch.empty([batch_size*mh_steps*n_aug]+list(aug_data.shape[1:]), device = aug_data.device, dtype= aug_data.dtype)
        ops = torch.empty([batch_size*mh_steps*n_aug], device = op.device, dtype= op.dtype)
        levels = torch.empty([batch_size*mh_steps*n_aug], device = level.device, dtype= level.dtype)
        proposals[0:batch_size] = aug_data
        ops[0:batch_size] = op
        levels[0:batch_size] = level
        ###########################################
        #   Compute Proposal Losses
        ###########################################
        for i in  range(1, mh_steps*n_aug):
            iter_loader = iter(aug_loader)
            aug_data, _ , op, level = next(iter_loader)
            aug_data = aug_data.to(f"cuda:{rank}")
            proposals[i*batch_size:(i+1)*batch_size] = aug_data
            ops[i*batch_size:(i+1)*batch_size] = op
            levels[i*batch_size:(i+1)*batch_size] = level
        with torch.no_grad():
            proposal_loss = cross_entropy(model(proposals), label.repeat(mh_steps*n_aug), reduction="none")
        ##################################
        #   Build MC chains
        ##################################
            proposals = proposals.reshape([mh_steps, n_aug]+list(aug_data.shape))
            mh_data = torch.empty_like(proposals)
            mh_data[0] = proposals[0]
            proposal_loss = proposal_loss.reshape([mh_steps,n_aug,  batch_size])
            mh_loss =  torch.empty_like(proposal_loss)
            mh_loss[0] = proposal_loss[0]
            ops = ops.reshape([mh_steps, n_aug, batch_size])
            levels = levels.reshape([mh_steps, n_aug, batch_size])
            mh_ops = torch.empty_like(ops)
            mh_levels = torch.empty_like(levels)
            mh_ops[0] = ops[0]
            mh_levels[0] = levels[0]
            last_loss = proposal_loss[0]
            ones = torch.ones_like(last_loss)
            accepted_all = torch.empty([mh_steps-1, n_aug, batch_size], device=last_loss.device, dtype=torch.bool)
            chain_state = mh_data[0]
            chain_op = ops[0]
            chain_level = levels[0]
            for i in  range(1, mh_steps):
                acceptance_ratio = torch.minimum(torch.nan_to_num(proposal_loss[i] / last_loss), ones)
                acceptance_ratio =  acceptance_ratio * (acceptance_ratio > 0)
                accepted = torch.bernoulli(acceptance_ratio).bool()
                accepted_all[i-1] = accepted
                #print("accepted", accepted.shape)
                #print("proposals", proposals.shape)
                #print("chain state", chain_state.shape)
                chain_state[accepted] = proposals[i][accepted]
                chain_op[accepted] = ops[i][accepted]
                chain_level[accepted] = levels[i][accepted]
                last_loss[accepted] = proposal_loss[i][accepted]
                mh_data[i] = chain_state
                mh_ops[i] = chain_op
                mh_levels[i] = chain_level
                mh_loss[i] = last_loss
            # Keep last Samples
            mh_data = mh_data[-1]
            mh_ops = mh_ops[-1]
            mh_levels = mh_levels[-1]
            mh_loss = mh_loss[-1]
            #print("mh",  mh_loss.shape)
            mh_data = torch.flatten(mh_data, start_dim=0, end_dim=1)
            mh_ops = torch.flatten(mh_ops, start_dim=0, end_dim=1)
            mh_levels = torch.flatten(mh_levels, start_dim=0, end_dim=1)
            aug_stats.update(mh_ops, mh_levels)

        ##############################
        #   Primal Descent Step
        ##############################
        if sample_constraint:
            if isinstance(dual_vars, torch.Tensor):
                clean = torch.bernoulli(torch.ones(data.shape[0])/(1+dual_vars.item())).bool()
            else:
                clean = torch.bernoulli(torch.ones(data.shape[0])/(1+dual_vars)).bool()
            mh_data[clean] = data[clean]
            preds = model(mh_data)
            loss = loss_fn(preds, label)
            mh_loss = mh_loss.mean()
        else:
            preds = model(data)
            clean_loss = loss_fn(preds, label)
            mh_preds = model(mh_data)
            mh_loss = loss_fn(mh_preds, label.repeat(n_aug))
            # Primal descent
            loss = clean_loss + dual_vars * mh_loss
        if optimizer:
            if communicate_grad:
                loss.backward()
            else:
                with model.no_sync():
                    loss.backward()

            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            if (steps-1) % C.get().get('step_optimizer_every', 1) == C.get().get('step_optimizer_nth_step', 0): # default is to step on the first step of each pack
                optimizer.step()
        ##############################
        #   Dual Ascent Step
        ##############################
        with torch.no_grad():
            dual_vars = relu(dual_vars + C.get()['PD'].get('lr') * (mh_loss.detach() - C.get()['PD'].get('margin')))
        ##############################
        #print(f"Time for forward/backward {time()-fb_time}")
        top1, top5 = accuracy(preds, label, (1, 5))
        if sample_constraint:
            metrics.add_dict({
                'loss': loss.item() * len(data),
                'aug loss': mh_loss.item() * len(data),
                'top1': top1.item() * len(data),
                'top5': top5.item() * len(data),
                'dual': dual_vars.item()* len(data),
                'acc ratio': accepted_all.float().mean().item()*len(data),
            })
        else:
            metrics.add_dict({
                'loss': clean_loss.item() * len(data),
                'aug loss': mh_loss.item() * len(data),
                'top1': top1.item() * len(data),
                'top5': top5.item() * len(data),
                'dual': dual_vars.item()* len(data),
                'acc ratio': accepted_all.float().mean().item()*len(data),
            })
        if steps % 2 == 0:
            metrics.add('eval_top1', top1.item() * len(data)) # times 2 since it is only recorded every sec step
            eval_cnt += len(data)
        cnt += len(data)
        if verbose:
            postfix = metrics.divide(cnt, eval_top1=eval_cnt)
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            logging_loader.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        #before_load_time = time()
        del preds, loss, top1, top5, data, label

    if tqdm_disable:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'],  metrics.divide(cnt, eval_top1=eval_cnt), optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics.divide(cnt, eval_top1=eval_cnt))

    metrics = metrics.divide(cnt, eval_top1=eval_cnt)
    if wandb_log:
        wandb.log(metrics.metrics)
        aug_stats.process()
        for op in aug_stats.transform_names:
            wandb.log({f"Transformation Levels {op}": wandb.Histogram(aug_stats.stats["levels"][op])})
            wandb.log({f"Transformation Counts {op}":aug_stats.stats["counts"][op]})

    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)

    return metrics, dual_vars


def train_and_eval(rank, worldsize, tag, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False, wandb_log = False, log_interval=40):
    if not reporter:
        reporter = lambda **kwargs: 0

    if not tag or (worldsize and torch.distributed.get_rank() > 0):
        from TrivialAugment.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided or rank > 0 -> no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter

    os.makedirs('./logs/', exist_ok=True)
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test', 'testtrain']]

    aug_lib.set_augmentation_space(C.get().get('augmentation_search_space', 'standard'), C.get().get('augmentation_parameter_max', 30), C.get().get('custom_search_space_augs', None))
    max_epoch = C.get()['epoch']
    trainsampler, trainloader, validloader, testloader_, testtrainloader_, dataset_info, augmented_dset = get_dataloaders(C.get()['dataset'], C.get()['batch'], dataroot, test_ratio, split_idx=cv_fold, distributed=worldsize>1, started_with_spawn=C.get()['started_with_spawn'], summary_writer=writers[0])

    # create a model & an optimizer
    model = get_model(C.get()['model'], C.get()['batch'], num_class(C.get()['dataset']), writer=writers[0])
    if worldsize > 1:
        model = DDP(model.to(rank), device_ids=[rank])
    else:
        model = model.to(f"cuda:{rank}")


    criterion = nn.CrossEntropyLoss()

    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=C.get()['optimizer']['nesterov']
        )
    elif C.get()['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
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
    result = OrderedDict()
    epoch_start = 1
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_path+=( f"/{C.get()['dataset']}_{C.get()['model']['type']}_{C.get()['aug']}_margin{C.get()['PD']['margin']}_seed{C.get()['seed']}.pkl")

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        with torch.no_grad():
            rs['train'] = run_epoch(rank, worldsize, model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0])
            rs['valid'] = run_epoch(rank, worldsize, model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1])
            rs['test'] = run_epoch(rank, worldsize, model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2])
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1 = 0
    if C.get()['n_aug']>1:
        dual_vars = 1
    else:
        dual_vars = 0
    for epoch in range(epoch_start, max_epoch + 1):
        if worldsize > 1:
            trainsampler.set_epoch(epoch)

        model.train()
        rs = dict()
        rs['train'], dual_vars = run_epoch_dual(rank, worldsize,model, trainloader, criterion, optimizer, dual_vars, wandb_log = wandb_log, augmented_dset=augmented_dset, desc_default='train', epoch=epoch, writer=writers[0], verbose=True, scheduler=scheduler, sample_pairing_loader=testtrainloader_)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch == max_epoch or epoch % log_interval == 0:
            with torch.no_grad():
                if C.get().get('compute_testtrain', False):
                    rs['testtrain'] = run_epoch(rank, worldsize, model, testtrainloader_, criterion, None, desc_default='testtrain', epoch=epoch, writer=writers[3], verbose=True)
                rs['test'] = run_epoch(rank, worldsize, model, testloader_, criterion, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=True)
                rs['valid'] = run_epoch(rank, worldsize, model, validloader, criterion, None, desc_default='valid', epoch=epoch, writer=writers[1], verbose=True)
                # save checkpoint
        early_finish_epoch = C.get().get('early_finish_epoch', None)
        if early_finish_epoch == epoch:
            break
        if wandb_log:
            if epoch == max_epoch:
                final_dict = {f"final {k}": v for k,v in result.items()}
                wandb.log(final_dict)
                if save_path and C.get().get('save_model', True):
                    #wandb.save()
                    print('save model@%d to %s' % (epoch, save_path))
                    torch.save(model.state_dict(), save_path)
                    artifact = wandb.Artifact('model', type='model')
                    artifact.add_file(save_path)
                    wandb.log_artifact(artifact)
            else:
                wandb.log({"train": rs["train"].get_dict(), "epoch":epoch, "dualvar": dual_vars})
                if epoch % log_interval == 0:
                    wandb.log({"test": rs["test"].get_dict(), "epoch":epoch})
                    wandb.log({"valid": rs["valid"].get_dict(), "epoch":epoch})


    del model

    return result

def setup(global_rank, local_rank, world_size, port_suffix):
    torch.cuda.set_device(local_rank)
    if port_suffix is not None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'12{port_suffix}'

        # initialize the process group
        dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
        return global_rank, world_size
    else:
        dist.init_process_group(backend='NCCL', init_method='env://')
        return torch.distributed.get_rank(), torch.distributed.get_world_size()

def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels',
                        help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--only-eval', action='store_true')
    parser.add_argument('--local_rank', default=None, type=int)
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--project', type=str, default='Daug-Gen', help='wandb project')
    return parser.parse_args()


def spawn_process(global_rank, worldsize, port_suffix, args, config_path=None, communicate_results_with_queue=None, local_rank=None,):
    if local_rank is None:
        local_rank = global_rank
    started_with_spawn = worldsize is not None and worldsize > 0
    if worldsize != 0:
        global_rank, worldsize = setup(global_rank, local_rank, worldsize, port_suffix)
    print('dist info', local_rank,global_rank,worldsize)
    #communicate_results_with_queue.value = 1.
    #return
    if args.config is not None:
        try:
            C(args.config[0])
            print("conf successfully loaded")
        except:
            print("theconf singleton error - ignore it")
    if args.wandb_log:
        print("logging")
        wandb.init(project=args.project, name=args.tag)
        wandb.config.update(args)
        wandb.config.update(C.get().flatten())
    C.get()['started_with_spawn'] = started_with_spawn

    if worldsize:
        assert worldsize == C.get()['gpus'], f"Did not specify the number of GPUs in Config with which it was started: {worldsize} vs {C.get()['gpus']}"
    else:
        assert 'gpus' not in C.get() or C.get()['gpus'] == 1

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    #if args.save:
        #add_filehandler(logger, args.save.replace('.pth', '.log'))

    #logger.info(json.dumps(C.get().conf, indent=4))
    if 'seed' in C.get():
        seed = C.get()['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False


    import time
    t = time.time()
    result = train_and_eval(local_rank, worldsize, args.tag, args.dataroot, wandb_log=args.wandb_log, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.save, only_eval=args.only_eval, metric='last')
    elapsed = time.time() - t
    print('done')

    logger.info(f'done on rank {global_rank}.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info(args.save)




@dataclass
class Args:
    tag: str = ''
    dataroot: str = None
    save: str = ''
    cv_ratio: float = 0.
    cv: int = 0
    only_eval: bool = False
    local_rank: None = None

def run_from_py(dataroot, config_dict, save=''):
    args = Args(dataroot=dataroot, save=save)
    with tempfile.NamedTemporaryFile(mode='w+') as f, tempfile.NamedTemporaryFile() as result_file:
        path = f.name
        yaml.dump(config_dict, f)
        world_size = torch.cuda.device_count()
        port_suffix = str(random.randint(100, 999))
        #result_queue = mp.get_context('spawn').Queue()
        result_queue = mp.get_context('spawn').Value('d',.0)
        if world_size > 1:
            outcome = mp.spawn(spawn_process,
                               args=(world_size, port_suffix, args, path, result_queue),
                               nprocs=world_size,
                               join=True)
        else:
            outcome = spawn_process(0, 0, port_suffix, args, path, result_queue)
        #result = result_queue.get()[0]
        result = result_queue.value
    return result


if __name__ == '__main__':
    pre_parser = ArgumentParser()
    pre_parser.add_argument('--local_rank', default=None, type=int)
    pre_parser.add_argument('--distributed', action='store_true')
    args, _ = pre_parser.parse_known_args()
    if args.local_rank is None:
        print("Spawning processes")
        world_size = torch.cuda.device_count()
        port_suffix = str(random.randint(10,99))
        if world_size > 1:
            outcome = mp.spawn(spawn_process,
                              args=(world_size,port_suffix,parse_args()),
                              nprocs=world_size,
                              join=True)
        else:
            spawn_process(0, 0, None, parse_args())       
    elif args.distributed:
        spawn_process(None, -1, None, parse_args(), local_rank=args.local_rank)
    else:
        spawn_process(2, 0, None, parse_args(), local_rank=args.local_rank)
