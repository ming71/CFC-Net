from __future__ import print_function

import os
import argparse
import numpy as np
import time
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.model import CFCNet
from eval import evaluate
from datasets import *
from utils.utils import *
from torch_warmup_lr import WarmupLR

mixed_precision = True
try:  
    from apex import amp
except:
    print('fail to speed up training via apex \n')
    mixed_precision = False  # not installed

DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'IC13': IC13Dataset,
            'HRSC2016': HRSCDataset,
            'DOTA':DOTADataset,
            'UCAS_AOD':UCAS_AODDataset,
            'NWPU_VHR':NWPUDataset
            }


def train_model(args, hyps):
    #  parse configs
    epochs = int(hyps['epochs'])
    batch_size = int(hyps['batch_size'])
    results_file = 'result.txt'
    weight =  'weights' + os.sep + 'last.pth' if args.resume or args.load else args.weight
    last = 'weights' + os.sep + 'last.pth'
    best = 'weights' + os.sep + 'best.pth'
    start_epoch = 0
    best_fitness = 0 #   max f1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # creat folder
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    for f in glob.glob(results_file):
        os.remove(f)

    # multi-scale
    if args.multi_scale:
        scales = args.training_size + 32 * np.array([x for x in range(-1, 5)])
        # set manually
        # scales = np.array([384, 480, 544, 608, 704, 800, 896, 960])
        print('Using multi-scale %g - %g' % (scales[0], scales[-1]))   
    else :
        scales = args.training_size 
############

    # dataloader
    assert args.dataset in DATASETS.keys(), 'Not supported dataset!'
    ds = DATASETS[args.dataset](dataset=args.train_path, augment=args.augment)
    collater = Collater(scales=scales, keep_ratio=True, multiple=32)
    loader = data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collater,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # Initialize model
    init_seeds()
    model = CFCNet(backbone=args.backbone, hyps=hyps)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyps['lr0'])
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler = WarmupLR(scheduler, init_lr=hyps['warmup_lr'], num_warmup=hyps['warm_epoch'], warmup_strategy='cos')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min = 1e-5)
    scheduler.last_epoch = start_epoch - 1
    ######## Plot lr schedule(注意只能测试lr时用！否则lr调整失效)#####
#     y = []
#     for _ in range(epochs):
#         scheduler.step()
#         y.append(optimizer.param_groups[0]['lr'])
#     plt.plot(y, label='LR')
#     plt.xlabel('epoch')
#     plt.ylabel('LR')
#     plt.tight_layout()
#     plt.savefig('LR.png', dpi=300)    
#     import ipdb; ipdb.set_trace()
    ###########################################

    # load chkpt
    if weight.endswith('.pth'):
        chkpt = torch.load(weight)
        # load model
        if 'model' in chkpt.keys() :
            model.load_state_dict(chkpt['model'])
        else:
            model.load_state_dict(chkpt)
        # load optimizer
        if 'optimizer' in chkpt.keys() and chkpt['optimizer'] is not None and args.resume :
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        # load results
        if 'training_results' in chkpt.keys() and  chkpt.get('training_results') is not None and args.resume:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt
        if args.resume and 'epoch' in chkpt.keys():
            start_epoch = chkpt['epoch'] + 1   

        del chkpt
 
    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
   
 
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    model_info(model, report='summary')  # 'full' or 'summary'
    # 'P', 'R', 'mAP', 'F1'
    results = (0, 0, 0, 0)

    for epoch in range(start_epoch,epochs):
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'ref', 'cls', 'reg', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(loader), total=len(loader))  # progress bar
        mloss = torch.zeros(3).cuda()
        for i, (ni, batch) in enumerate(pbar):
            
            model.train()

            if args.freeze_bn:
                if torch.cuda.device_count() > 1:
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()

            optimizer.zero_grad()
            ims, gt_boxes = batch['image'], batch['boxes']
            if torch.cuda.is_available():
                ims, gt_boxes = ims.cuda(), gt_boxes.cuda()
            losses = model(ims, gt_boxes,process =epoch/epochs )
            loss_cls, loss_reg = losses['loss_cls'].mean(), losses['loss_reg'].mean()
            if losses.__contains__('loss_ref'):
                loss_ref = losses['loss_ref'].mean()
                # 注意这里又将最终的reg和ref取均值
                loss = loss_cls + (loss_reg + loss_ref)*0.5 
            else:
                loss = loss_cls + loss_reg
            if not torch.isfinite(loss):
                import ipdb; ipdb.set_trace()
                print('WARNING: non-finite loss, ending training ')
                break
            if bool(loss == 0):
                continue

            # calculate gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            # Print batch results
            if losses.__contains__('loss_ref'):
                loss_items = torch.stack([loss_ref, loss_cls, loss_reg], 0).detach()
            else: 
                loss_items = torch.stack([torch.zeros_like(loss_cls),loss_cls, loss_reg], 0).detach()
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)

            s = ('%10s' * 2 + '%10.3g' * 6) % (
                  '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, mloss.sum(), gt_boxes.shape[1], min(ims.shape[2:]))
            pbar.set_description(s)

        # Update scheduler
        scheduler.step()
        final_epoch = epoch + 1 == epochs
        
        # eval
        if hyps['test_interval']!= -1 and epoch % hyps['test_interval'] == 0 and epoch > 10 :
            if torch.cuda.device_count() > 1:
                results = evaluate(target_size=args.target_size,
                                   test_path=args.test_path,
                                   dataset=args.dataset,
                                   model=model.module, 
                                   hyps=hyps,
                                   conf = 0.01 if final_epoch else 0.1)    
            else:
                results = evaluate(target_size=args.target_size,
                                   test_path=args.test_path,
                                   dataset=args.dataset,
                                   model=model,
                                   hyps=hyps,
                                   conf = 0.01 if final_epoch else 0.1) #  p, r, map, f1

        
        # Write result log
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 4 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        ##   Checkpoint
        if arg.dataset in ['IC15', ['IC13']]:
            fitness = results[-1]   # Update best f1
        else :
            fitness = results[-2]   # Update best mAP
        if fitness > best_fitness:
            best_fitness = fitness

        with open(results_file, 'r') as f:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_fitness': best_fitness,
                     'training_results': f.read(),
                     'model': model.module.state_dict() if type(
                        model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': None if final_epoch else optimizer.state_dict()}
        

        # Save last checkpoint
        torch.save(chkpt, last)
        # Save best checkpoint
        if best_fitness == fitness:
            torch.save(chkpt, best) 

        if (epoch % hyps['save_interval'] == 0  and epoch > 10) or final_epoch:
            if torch.cuda.device_count() > 1:
                torch.save(chkpt, './weights/deploy%g.pth'% epoch)
            else:
                torch.save(chkpt, './weights/deploy%g.pth'% epoch)

    # end training
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a detector')
    # config
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    # network
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--freeze_bn', type=bool, default=False)
    parser.add_argument('--weight', type=str, default='')   # 
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    
     # HRSC2016
    parser.add_argument('--dataset', type=str, default='HRSC2016')      
    parser.add_argument('--train_path', type=str, default='/py/datasets/HRSC2016/train.txt')    
    parser.add_argument('--test_path', type=str, default='/py/datasets/HRSC2016/test.txt')        

    # DOTA
    # parser.add_argument('--dataset', type=str, default='DOTA')    
    # parser.add_argument('--train_path', type=str, default='DOTA/trainval.txt')

    # IC15
    # parser.add_argument('--dataset', type=str, default='IC15')
    # parser.add_argument('--train_path', type=str, default='ICDAR15/train.txt')
    # parser.add_argument('--test_path', type=str, default='ICDAR15/test')

    # IC13
    # parser.add_argument('--dataset', type=str, default='IC13')
    # parser.add_argument('--train_path', type=str, default='ICDAR13/train.txt')
    # parser.add_argument('--test_path', type=str, default='ICDAR13/test')

    # UCAS-AOD
#     parser.add_argument('--dataset', type=str, default='UCAS_AOD')
#     parser.add_argument('--train_path', type=str, default='UCAS_AOD/train.txt')
#     parser.add_argument('--test_path', type=str, default='UCAS_AOD/test.txt')

    # VOC2007
#     parser.add_argument('--dataset', type=str, default='VOC')
#     parser.add_argument('--train_path', type=str, default='VOC2007/ImageSets/Main/trainval.txt')
#     parser.add_argument('--test_path', type=str, default='VOC2007/ImageSets/Main/test.txt')

    # NWPU-VHR10
    # parser.add_argument('--dataset', type=str, default='NWPU_VHR')
    # parser.add_argument('--train_path', type=str, default='NWPU_VHR/train.txt')
    # parser.add_argument('--test_path', type=str, default='NWPU_VHR/test.txt')

    parser.add_argument('--training_size', type=int, default=416)
    parser.add_argument('--resume', action='store_true', help='resume training from last.pth')
    parser.add_argument('--load', action='store_true', help='load training from last.pth')
    parser.add_argument('--augment', action='store_true', help='data augment')
    parser.add_argument('--target_size', type=int, default=[416])   # 支持多尺度测试
    #

    arg = parser.parse_args()
    hyps = hyp_parse(arg.hyp)
    print(arg)
    print(hyps)

    train_model(arg, hyps)