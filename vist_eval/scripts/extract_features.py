from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import splitext
import sys
import json
import argparse
import string
import glob
import time

import h5py
from six.moves import cPickle
import numpy as np
from scipy.misc import imread

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms as trn
from misc.resnet_utils import myResnet
import misc.resnet as resnet

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

types = ['.jpg', '.gif', '.png']


def main(params):
    assert params['feature_type'] in ['fc', 'conv', 'both']
    compute_fc = params['feature_type'] == 'fc' or params['feature_type'] == 'both'
    compute_conv = params['feature_type'] == 'conv' or params['feature_type'] == 'both'

    net = getattr(resnet, params['model'])()
    net.load_state_dict(torch.load(os.path.join(params['model_root'], params['model'] + '.pth')))
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()

    if compute_fc:
        dir_fc = os.path.join(params['out_dir'], 'fc')
        if not os.path.exists(dir_fc):
            os.makedirs(dir_fc)
    if compute_conv:
        dir_conv = os.path.join(params['out_dir'], 'conv')
        if not os.path.exists(dir_conv):
            os.makedirs(dir_conv)

    for split in ['train', 'val', 'test']:
        count = 0
        if compute_fc and not os.path.exists(os.path.join(dir_fc, split)):
            os.makedirs(os.path.join(dir_fc, split))
        if compute_conv and not os.path.exists(os.path.join(dir_conv, split)):
            os.makedirs(os.path.join(dir_conv, split))

        files = glob.glob("{}/{}/*.jpg".format(params['img_dir'], split))
        start = time.time()
        for file in files:
            count += 1
            basename = os.path.basename(file)
            img_id = splitext(basename)[0]
            try:
                I = imread(file)
            except:
                I = np.zeros((224, 224, 3), 'float32')

            # handle grayscale input frames
            if len(I.shape) == 2:
                I = I[:, :, np.newaxis]
                I = np.concatenate((I, I, I), axis=2)

            I = I.astype('float32') / 255.0
            I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
            I = Variable(preprocess(I), volatile=True)
            tmp_fc, tmp_conv = my_resnet(I, params['att_size'])

            # write to pkl
            if compute_fc:
                np.save(os.path.join(dir_fc, split, img_id), tmp_fc.data.cpu().float().numpy())
            if compute_conv:
                np.savez_compressed(os.path.join(dir_conv, split, img_id), tmp_conv.data.cpu().float().numpy())

            if count % 100 == 0:
                print('processing {} set -- {}/{} {:.3}%, time used: {}s'.format(split, count, len(files),
                                                                                 count * 100.0 / len(files),
                                                                                 time.time() - start))
                start = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type=str, default='/mnt/sshd/xwang/VIST/images_256')
    parser.add_argument('--out_dir', type=str, default='/mnt/sshd/xwang/VIST/resnet_features')
    parser.add_argument('--feature_type', type=str, default='fc', help='fc, conv, both')

    parser.add_argument('--model', type=str, default='resnet152', help='resnet101, resnet152')
    parser.add_argument('--model_root', default='./data/imagenet_weights', type=str, help='model root')
    parser.add_argument('--att_size', type=int, default=7,
                        help='14: 14x14 conv feature, 7: 7x7 conv feature')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
