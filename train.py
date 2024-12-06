import torch
from torch.autograd import Variable
import os
import torch.nn as nn
import argparse
from datetime import datetime
from lib.MLDNet import MLDNet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter,adjust_lr_d
import torch.nn.functional as F
import numpy as np
import logging
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from utils.loss import CrossEntropyWithL1


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice



def structure_loss(pred, mask):

    CE = CrossEntropyWithL1(mode='binary')
    CELOSS = CE(pred,mask)

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (CELOSS + wiou).mean()
#############################################################################################################


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    #print(f'总计：{num1}')
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
   
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1, res2, res3,res4 = model(image)
        # res1, res2, res3, res4 = model(image)
        # eval Dice
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        # res1 = F.interpolate(res1, size=gt.shape, mode='bilinear', align_corners=False)
        # res1 = res1.sigmoid().data.cpu().numpy().squeeze()
        # res1 = (res1 - res1.min()) / (res1.max() - res1.min() + 1e-8)
        # input = res1
        # target = np.array(gt)
        
        mean_dice=mean_dice_np(target,input)
        DSC=DSC+mean_dice

   
    return DSC / num1


def train(train_loader, model, optimizer, epoch, test_path, max_steps=None):
    model.train()
    global best,Best_loss
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    total_step = len(train_loader) if not max_steps else max_steps
    for i, pack in enumerate(train_loader, start=1):
        if i > total_step:
            break
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2, P3, P4,P5= model(images)
            # P2, P3, P4, P5 = model(images)

            weights = [1.0,1.0,1.0,1.0,1.0]
            
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)*weights[0]
            # loss_P1 = 0.0
            loss_P2 = structure_loss(P2, gts)*weights[1]
            loss_P3 = structure_loss(P3, gts)*weights[2]
            loss_P4 = structure_loss(P4, gts)*weights[3]
            loss_P5 = structure_loss(P5, gts)*weights[4]

            
           
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4 + loss_P5

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch[{:03d}/{:03d}], Step[{:04d}/{:04d}],'
                  ' loss-all:[{:0.4f}], P1:[{:0.4f}] lr:[{:0.7f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(),loss_P1, optimizer.param_groups[0]['lr']))
            logging.info('{} Epoch[{:03d}/{:03d}], Step[{:04d}/{:04d}],'
                  ' loss-all:[{:0.4f}], P1:[{:0.4f}] lr:[{:0.7f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(),loss_P1, optimizer.param_groups[0]['lr']))
    # save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    global dict_plot

    if (epoch + 1) % 25 == 0:
        meandice_list = []  # 用于存储本轮次所有数据集的Dice系数

        # 直接在每个epoch后测试五个数据集
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            meandice = test(model, test_path, dataset)
            meandice_list.append(meandice)
           # print(f'{dataset} meandice: {meandice}）
            dict_plot[dataset].append(meandice)

        # 计算本轮次的平均Dice系数
        meandice_avg = sum(meandice_list) / len(meandice_list)
        print(f'Average meandice for this epoch: {meandice_avg}')

        # 更新最好的Dice系数并保存模型
        if meandice_avg > best:
            best = meandice_avg
            print('#' * 80)
            print(f'New best meandice: {best}')
            print('#' * 80)
            torch.save(model.state_dict(), save_path + 'testopen.pth')
            for dataset, meandice in zip(['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB'],
                                         meandice_list):
                print(f'{dataset} meandice: {meandice}')
                dict_plot[dataset].append(meandice)

        print('#' * 80)

if __name__ == '__main__':

    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=150, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.5, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                       default='./data/TrainDatasetEdge/',
                       help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./data/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/polyp/')
    parser.add_argument('--log_path', type=str,default='./log/')

    opt = parser.parse_args()

    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)
    logging.basicConfig(filename=f'{opt.log_path}train_log_{int(time.time())}.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    
    model = MLDNet().cuda()

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    logging.info(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-5)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Trainng", "#" * 20)

    for epoch in range(1, opt.epoch):

        train(train_loader, model, optimizer, epoch, opt.test_path,max_steps=20)
        scheduler.step()

