from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import logging
import time
import numpy as np

from .model_utils import AttentionLayer, VisualEncoder, _smallest


def from_numpy(states):
    return [Variable(torch.from_numpy(state)).cuda() for state in states]


class RewardModel(nn.Module):
    def __init__(self, opt):
        super(RewardModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.word_embed_dim = 300
        self.feat_size = opt.feat_size
        self.kernel_num = 512
        self.kernels = [2, 3, 4, 5]
        self.out_dim = len(self.kernels) * self.kernel_num + self.word_embed_dim

        self.emb = nn.Embedding(self.vocab_size, self.word_embed_dim)
        self.emb.weight.data.copy_(torch.from_numpy(np.load("VIST/embedding.npy")))

        self.proj = nn.Linear(self.feat_size, self.word_embed_dim)

        self.convs = [nn.Conv2d(1, self.kernel_num, (k, self.word_embed_dim)) for k in self.kernels]

        self.dropout = nn.Dropout(opt.dropout)

        self.fc = nn.Linear(self.out_dim, 1, bias=True)

        if opt.activation.lower() == "linear":
            self.activation = None
        elif opt.activation.lower() == "sign":
            self.activation = nn.Softsign()
        elif self.activation.lower() == "tahn":
            self.activation = nn.Tanh()

    def forward(self, story, feature):
        embedding = Variable(self.emb(story).data)  # (batch_size, seq_length, embed_dim)

        self.convs = [model.cuda() for model in self.convs]

        # batch x seq_len x emb_dim -> batch x 1 x seq_len x emb_dim
        embedding = embedding.unsqueeze(1)
        x = [F.relu(conv(embedding)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        # combine with image feature
        img = self.proj(feature)
        combined = torch.cat([x, img], 1)
        combined = self.dropout(combined)

        prob = self.fc(combined).view(-1)
        return self.activation(prob)
