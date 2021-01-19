from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
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
from criterion import to_contiguous
from misc.yellowfin import YFOptimizer
from train import setup_optimizer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Flag:
    def __init__(self, D_iters, G_iters, always=None):
        self.D_iters = D_iters
        self.G_iters = G_iters

        self.flag = "Disc"
        self.iters = self.D_iters
        self.curr = 0
        self.always = always

    def inc(self):
        self.curr += 1
        if self.curr >= self.iters and self.always is None:
            if self.flag == "Disc":
                self.flag = "Gen"
                self.iters = self.G_iters
            elif self.flag == "Gen":
                self.flag = "Disc"
                self.iters = self.D_iters
            self.curr = 0


def train(opt):
    logger = Logger(opt)
    flag = Flag(D_iters=opt.D_iter, G_iters=opt.G_iter, always=opt.always)
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
    rl_crit = criterion.ReinforceCriterion(opt, dataset)

    # set up model
    model = models.setup(opt)
    model.cuda()
    disc_opt = copy.copy(opt)
    disc_opt.model = 'RewardModel'
    disc = models.setup(disc_opt)
    if os.path.exists(os.path.join(logger.log_dir, 'disc-model.pth')):
        logging.info("loading pretrained RewardModel")
        disc.load_state_dict(torch.load(os.path.join(logger.log_dir, 'disc-model.pth')))
    disc.cuda()

    # set up optimizer
    optimizer = setup_optimizer(opt, model)
    disc_optimizer = setup_optimizer(opt, disc)

    dataset.train()
    model.train()
    disc.train()
    ############################## training ##################################
    for epoch in range(logger.epoch_start, opt.max_epochs):
        # Assign the scheduled sampling prob

        start = time.time()
        for iter, batch in enumerate(train_loader):
            logger.iteration += 1
            torch.cuda.synchronize()

            feature_fc = Variable(batch['feature_fc']).cuda()
            target = Variable(batch['split_story']).cuda()
            index = batch['index']

            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            if flag.flag == "Disc":
                model.eval()
                disc.train()
                if opt.decoding_method_DISC == 'sample':
                    seq, seq_log_probs, baseline = model.sample(feature_fc, sample_max=False, rl_training=True,
                                                                pad=True)
                elif opt.decoding_method_DISC == 'greedy':
                    seq, seq_log_probs, baseline = model.sample(feature_fc, sample_max=True, rl_training=True,
                                                                pad=True)
            else:
                model.train()
                disc.eval()
                seq, seq_log_probs, baseline = model.sample(feature_fc, sample_max=False, rl_training=True, pad=True)

            seq = Variable(seq).cuda()
            mask = (seq > 0).float()
            mask = to_contiguous(
                torch.cat([Variable(mask.data.new(mask.size(0), mask.size(1), 1).fill_(1)), mask[:, :, :-1]], 2))
            normed_seq_log_probs = (seq_log_probs * mask).sum(-1) / mask.sum(-1)

            gen_score = disc(seq.view(-1, seq.size(2)), feature_fc.view(-1, feature_fc.size(2)))

            if flag.flag == "Disc":
                gt_score = disc(target.view(-1, target.size(2)), feature_fc.view(-1, feature_fc.size(2)))
                loss = -torch.sum(gt_score) + torch.sum(gen_score)

                avg_pos_score = torch.mean(gt_score)
                avg_neg_score = torch.mean(gen_score)

                if logger.iteration % 5 == 0:
                    logging.info("pos reward {} neg reward {}".format(avg_pos_score.data[0], avg_neg_score.data[0]))
                    print("PREDICTION: ", utils.decode_story(dataset.get_vocab(), seq[:1].data)[0])
                    print("GROUND TRUTH: ", utils.decode_story(dataset.get_vocab(), target[:1].data)[0])
            else:
                rewards = Variable(gen_score.data - 0 * normed_seq_log_probs.data)
                #with open("/tmp/reward.txt", "a") as f:
                #    print(" ".join(map(str, rewards.data.cpu().numpy())), file=f)
                loss, avg_score = rl_crit(seq.data, seq_log_probs, baseline, index, rewards)
                # if logger.iteration % opt.losses_log_every == 0:
                avg_pos_score = torch.mean(gen_score)
                logging.info(
                    "average reward: {} average IRL score: {}".format(avg_score.data[0], avg_pos_score.data[0]))

            if flag.flag == "Disc":
                loss.backward()
                nn.utils.clip_grad_norm(disc.parameters(), opt.grad_clip, norm_type=2)
                disc_optimizer.step()
            else:
                tf_loss = crit(model(feature_fc, target), target)
                print("rl_loss / tf_loss = ", loss.data[0] / tf_loss.data[0])
                loss = opt.rl_weight * loss + (1 - opt.rl_weight) * tf_loss
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip, norm_type=2)
                optimizer.step()

            train_loss = loss.data[0]
            torch.cuda.synchronize()

            # Write the training loss summary
            if logger.iteration % opt.losses_log_every == 0:
                logger.log_training(epoch, iter, train_loss, opt.learning_rate, model.ss_prob)
                logging.info(
                    "Epoch {} Train {} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(epoch, flag.flag,
                                                                                                  iter,
                                                                                                  len(train_loader),
                                                                                                  train_loss,
                                                                                                  time.time() - start))
                start = time.time()

            if logger.iteration % opt.save_checkpoint_every == 0:
                if opt.always is None:
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
                        if bad_valid >= 10:
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
                else:
                    torch.save(disc.state_dict(), os.path.join(logger.log_dir, 'disc-model.pth'))
            flag.inc()


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
