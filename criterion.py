import collections
import time
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import logging
from vist_eval.meteor.meteor import Meteor
import misc.utils as utils


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class ReinforceCriterion(nn.Module):
    def __init__(self, opt, dataset):
        super(ReinforceCriterion, self).__init__()
        self.dataset = dataset
        self.reward_type = opt.reward_type
        self.bleu = None

        if self.reward_type == 'METEOR':
            from vist_eval.meteor.meteor import Meteor
            self.reward_scorer = Meteor()
        elif self.reward_type == 'CIDEr':
            sys.path.append("cider")
            from pyciderevalcap.ciderD.ciderD import CiderD
            self.reward_scorer = CiderD(df=opt.cached_tokens)
        elif self.reward_type == 'Bleu_4' or self.reward_type == 'Bleu_3':
            from vist_eval.bleu.bleu import Bleu
            self.reward_scorer = Bleu(4)
            self.bleu = int(self.reward_type[-1]) - 1
        elif self.reward_type == 'ROUGE_L':
            from vist_eval.rouge.rouge import Rouge
            self.reward_scorer = Rouge()
        else:
            err_msg = "{} scorer hasn't been implemented".format(self.reward_type)
            logging.error(err_msg)
            raise Exception(err_msg)

    def _cal_action_loss(self, log_probs, reward, mask):
        output = - log_probs * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

    def _cal_value_loss(self, reward, baseline, mask):
        output = (reward - baseline).pow(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

    def forward(self, seq, seq_log_probs, baseline, index, rewards=None):
        '''
        :param seq: (batch_size, 5, seq_length)
        :param seq_log_probs: (batch_size, 5, seq_length)
        :param baseline: (batch_size, 5, seq_length)
        :param indexes: (batch_size,)
        :param rewards: (batch_size, 5, seq_length)
        :return:
        '''
        if rewards is None:
            # compute the reward
            sents = utils.decode_story(self.dataset.get_vocab(), seq)

            rewards = []
            batch_size = seq.size(0)
            for i, story in enumerate(sents):
                vid, _ = self.dataset.get_id(index[i])
                GT_story = self.dataset.get_GT(index[i])
                result = {vid: [story]}
                gt = {vid: [GT_story]}
                score, _ = self.reward_scorer.compute_score(gt, result)
                if self.bleu is not None:
                    rewards.append(score[self.bleu])
                else:
                    rewards.append(score)
            rewards = torch.FloatTensor(rewards)  # (batch_size,)
            avg_reward = rewards.mean()
            rewards = Variable(rewards.view(batch_size, 1, 1).expand_as(seq)).cuda()
        else:
            avg_reward = rewards.mean()
            rewards = rewards.view(-1, 5, 1)

        # get the mask
        mask = (seq > 0).float()  # its size is supposed to be (batch_size, 5, seq_length)
        if mask.size(2) > 1:
            mask = torch.cat([mask.new(mask.size(0), mask.size(1), 1).fill_(1), mask[:, :, :-1]], 2).contiguous()
        else:
            mask.fill_(1)
        mask = Variable(mask)

        # compute the loss
        advantage = Variable(rewards.data - baseline.data)
        value_loss = self._cal_value_loss(rewards, baseline, mask)
        action_loss = self._cal_action_loss(seq_log_probs, advantage, mask)

        return action_loss + value_loss, avg_reward


class LanguageModelCriterion(nn.Module):
    def __init__(self, weight=0.0):
        self.weight = weight
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, weights=None, compute_prob=False):
        if len(target.size()) == 3:  # separate story
            input = input.view(-1, input.size(2), input.size(3))
            target = target.view(-1, target.size(2))

        seq_length = input.size(1)
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = (target > 0).float()
        mask = to_contiguous(torch.cat([Variable(mask.data.new(mask.size(0), 1).fill_(1)), mask[:, :-1]], 1))

        # reshape the variables
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = mask.view(-1, 1)

        if weights is None:
            output = - input.gather(1, target) * mask
        else:
            output = - input.gather(1, target) * mask * to_contiguous(weights).view(-1, 1)

        if compute_prob:
            output = output.view(-1, seq_length)
            mask = mask.view(-1, seq_length)
            return output.sum(-1) / mask.sum(-1)

        output = torch.sum(output) / torch.sum(mask)

        entropy = -(torch.exp(input) * input).sum(-1) * mask
        entropy = torch.sum(entropy) / torch.sum(mask)

        return output + self.weight * entropy
