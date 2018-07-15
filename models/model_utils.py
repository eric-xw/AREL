from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
import time


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim_en, hidden_dim_de, projected_size):
        super(AttentionLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_dim_en, projected_size)
        self.linear2 = nn.Linear(hidden_dim_de, projected_size)
        self.linear3 = nn.Linear(projected_size, 1, False)

    def forward(self, out_e, h):
        '''
        out_e: batch_size * num_frames * en_hidden_dim
        h : batch_size * de_hidden_dim
        '''
        assert out_e.size(0) == h.size(0)
        batch_size, num_frames, _ = out_e.size()
        hidden_dim = h.size(1)

        h_att = h.unsqueeze(1).expand(batch_size, num_frames, hidden_dim)
        x = F.tanh(F.dropout(self.linear1(out_e)) + F.dropout(self.linear2(h_att)))
        x = F.dropout(self.linear3(x))
        a = F.softmax(x.squeeze(2))

        return a


def _smallest(matrix, k, only_first_row=False):
    if only_first_row:
        flatten = matrix[:1, :].flatten()
    else:
        flatten = matrix.flatten()
    args = np.argpartition(flatten, k)[:k]
    args = args[np.argsort(flatten[args])]
    return np.unravel_index(args, matrix.shape), flatten[args]


class VisualEncoder(nn.Module):
    def __init__(self, opt):
        super(VisualEncoder, self).__init__()
        # embedding (input) layer options
        self.feat_size = opt.feat_size
        self.embed_dim = opt.word_embed_dim
        # rnn layer options
        self.rnn_type = opt.rnn_type
        self.num_layers = opt.num_layers
        self.hidden_dim = opt.hidden_dim
        self.dropout = opt.visual_dropout
        self.story_size = opt.story_size
        self.with_position = opt.with_position

        # visual embedding layer
        self.visual_emb = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
                                        nn.BatchNorm1d(self.embed_dim),
                                        nn.ReLU(True))

        # visual rnn layer
        self.hin_dropout_layer = nn.Dropout(self.dropout)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                              dropout=self.dropout, batch_first=True, bidirectional=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                               dropout=self.dropout, batch_first=True, bidirectional=True)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        # residual part
        self.project_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.relu = nn.ReLU()

        if self.with_position:
            self.position_embed = nn.Embedding(self.story_size, self.embed_dim)

    def init_hidden(self, batch_size, bi, dim):
        # the first parameter from the class
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return Variable(weight.new(self.num_layers * times, batch_size, dim).zero_())
        else:
            return (Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()),
                    Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()))

    def forward(self, input, hidden=None):
        """
        inputs:
        - input  	(batch_size, 5, feat_size)
        - hidden 	(num_layers * num_dirs, batch_size, hidden_dim // 2)
        return:
        - out 		(batch_size, 5, rnn_size), serve as context
        """
        batch_size, seq_length = input.size(0), input.size(1)

        # visual embeded
        emb = self.visual_emb(input.view(-1, self.feat_size))
        emb = emb.view(batch_size, seq_length, -1)  # (Na, album_size, embedding_size)

        # visual rnn layer
        if hidden is None:
            hidden = self.init_hidden(batch_size, bi=True, dim=self.hidden_dim // 2)
        rnn_input = self.hin_dropout_layer(emb)  # apply dropout
        houts, hidden = self.rnn(rnn_input, hidden)

        # residual layer
        out = emb + self.project_layer(houts)
        out = self.relu(out)  # (batch_size, 5, embed_dim)

        if self.with_position:
            for i in xrange(self.story_size):
                position = Variable(input.data.new(batch_size).long().fill_(i))
                out[:, i, :] = out[:, i, :] + self.position_embed(position)

        return out, hidden
