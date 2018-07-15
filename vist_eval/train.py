from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
import time
import sys
import logging

import opts
from dataset import VISTDataset
import models
from log_utils import Logger
import misc.utils as utils

from eval_utils import Evaluator
import criterion
from misc.yellowfin import YFOptimizer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def setup_optimizer(opt, model):
    if opt.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=opt.learning_rate,
                               betas=(opt.optim_alpha, opt.optim_beta),
                               eps=opt.optim_epsilon,
                               weight_decay=opt.weight_decay)
    elif opt.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=opt.learning_rate,
                              momentum=opt.momentum)
    elif opt.optim == "momSGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=opt.learning_rate,
                                    momentum=opt.momentum)
    elif opt.optim == 'Adadelta':
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=opt.learning_rate,
                                   weight_decay=opt.weight_decay)
    elif opt.optim == 'RMSprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    else:
        logging.error("Unknown optimizer: {}".format(opt.optim))
        raise Exception("Unknown optimizer: {}".format(opt.optim))

    # Load the optimizer
    if opt.resume_from is not None:
        optim_path = os.path.join(opt.resume_from, "optimizer.pth")
        if os.path.isfile(optim_path):
            logging.info("Load optimizer from {}".format(optim_path))
            optimizer.load_state_dict(torch.load(optim_path))
            opt.learning_rate = optimizer.param_groups[0]['lr']
            logging.info("Loaded learning rate is {}".format(opt.learning_rate))

    return optimizer


def train(opt):
    logger = Logger(opt)

    ################### set up dataset and dataloader ########################
    dataset = VISTDataset(opt)
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.get_story_length()

    dataset.set_option(data_type={'whole_story': False, 'split_story': True, 'caption': False})

    dataset.train()
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)
    dataset.val()
    val_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    ##################### set up model, criterion and optimizer ######
    bad_valid = 0

    # set up evaluator
    evaluator = Evaluator(opt, 'val')

    # set up criterion
    crit = criterion.LanguageModelCriterion()
    if opt.start_rl >= 0:
        rl_crit = criterion.ReinforceCriterion(opt, dataset)

    # set up model
    model = models.setup(opt)
    model.cuda()

    # set up optimizer
    optimizer = setup_optimizer(opt, model)

    dataset.train()
    model.train()
    ############################## training ##################################
    for epoch in range(logger.epoch_start, opt.max_epochs):
        # Assign the scheduled sampling prob
        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob *
                              frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob

        for iter, batch in enumerate(train_loader):
            start = time.time()
            logger.iteration += 1
            torch.cuda.synchronize()

            feature_fc = Variable(batch['feature_fc']).cuda()
            target = Variable(batch['split_story']).cuda()
            index = batch['index']

            optimizer.zero_grad()

            # cross entropy loss
            output = model(feature_fc, target)
            loss = crit(output, target)

            if opt.start_rl >= 0 and epoch >= opt.start_rl:  # reinforcement learning
                seq, seq_log_probs, baseline = model.sample(feature_fc, sample_max=False, rl_training=True)
                rl_loss, avg_score = rl_crit(seq, seq_log_probs, baseline, index)
                print(rl_loss.data[0] / loss.data[0])
                loss = opt.rl_weight * rl_loss + (1 - opt.rl_weight) * loss
                logging.info("average {} score: {}".format(opt.reward_type, avg_score))

            loss.backward()
            train_loss = loss.data[0]

            nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip, norm_type=2)
            optimizer.step()
            torch.cuda.synchronize()

            logging.info("Epoch {} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(epoch, iter,
                                                                                              len(train_loader),
                                                                                              train_loss,
                                                                                              time.time() - start))
            # Write the training loss summary
            if logger.iteration % opt.losses_log_every == 0:
                logger.log_training(epoch, iter, train_loss, opt.learning_rate, model.ss_prob)

            if logger.iteration % opt.save_checkpoint_every == 0:
                # Evaluate on validation dataset and save model for every epoch
                val_loss, predictions, metrics = evaluator.eval_story(model, crit, dataset, val_loader, opt)
                if opt.metric == 'XE':
                    score = -val_loss
                else:
                    score = metrics[opt.metric]
                logger.log_checkpoint(epoch, val_loss, metrics, predictions, opt, model, dataset, optimizer)
                # halve the learning rate if not improving for a long time
                if logger.best_val_score > score:
                    bad_valid += 1
                    if bad_valid >= 4:
                        opt.learning_rate = opt.learning_rate / 2.0
                        logging.info("halve learning rate to {}".format(opt.learning_rate))
                        checkpoint_path = os.path.join(logger.log_dir, 'model-best.pth')
                        model.load_state_dict(torch.load(checkpoint_path))
                        utils.set_lr(optimizer, opt.learning_rate)  # set the decayed rate
                        bad_valid = 0
                        logging.info("bad valid : {}".format(bad_valid))
                else:
                    logging.info("achieving best {} score: {}".format(opt.metric, score))
                    bad_valid = 0


def test(opt):
    logger = Logger(opt)
    dataset = VISTDataset(opt)
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.get_story_length()

    dataset.test()
    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    evaluator = Evaluator(opt, 'test')
    model = models.setup(opt)
    model.cuda()
    predictions, metrics = evaluator.test_story(model, dataset, test_loader, opt)


if __name__ == "__main__":
    opt = opts.parse_opt()

    if opt.option == 'train':
        print('Begin training:')
        train(opt)
    else:
        print('Begin testing:')
        test(opt)
