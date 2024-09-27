import argparse
import logging
import os
import random
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config_setting_synapse import setting_config
from models.cavmunet.CaVMamba import CaVMamba
from utils import test_single_volume


def inference(args, model, test_save_path=None):
    db_test = args.datasets(base_dir='', split="test_vol", list_dir='')

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.input_size_h, args.input_size_h],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class (%d)  mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jacard = np.mean(metric_list, axis=0)[2]
    mean_asd = np.mean(metric_list, axis=0)[3]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
    return "Testing Finished!"


if __name__ == "__main__":

    args = setting_config()
    if not args.distributed:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    snapshot_path = ''

    net = CaVMamba(
        num_classes=9,
        input_channels=3,
        depths=[2, 2, 2, 1],
        depths_decoder=[1, 2, 2, 2],
        drop_path_rate=0.2,
        load_ckpt_path='./pre_trained_weights/vmamba_small_e238_ema.pth',
    )

    net = net.cuda()

    net.load_state_dict(torch.load(snapshot_path),strict=False)
    # net.load_state_dict(torch.load(snapshot_path))

    if args.test_weights_path:

        test_save_path = os.path.join('./test')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


