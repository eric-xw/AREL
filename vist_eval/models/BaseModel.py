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


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.story_size = opt.story_size
        self.word_embed_dim = opt.word_embed_dim
        self.hidden_dim = opt.hidden_dim
        self.num_layers = opt.num_layers
        self.rnn_type = opt.rnn_type
        self.dropout = opt.dropout
        self.seq_length = opt.seq_length
        self.feat_size = opt.feat_size
        self.decoder_input_dim = self.word_embed_dim + self.word_embed_dim
        self.ss_prob = 0.0  # Schedule sampling probability

        # Visual Encoder
        self.encoder = VisualEncoder(opt)

        # Decoder LSTM
        self.project_d = nn.Linear(self.decoder_input_dim, self.word_embed_dim)
        if self.rnn_type == 'gru':
            self.decoder = nn.GRU(input_size=self.word_embed_dim, hidden_size=self.hidden_dim, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.decoder = nn.LSTM(input_size=self.word_embed_dim, hidden_size=self.hidden_dim, batch_first=True)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        # word embedding layer
        self.embed = nn.Embedding(self.vocab_size, self.word_embed_dim)

        # last linear layer
        self.logit = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                                   nn.Tanh(),
                                   nn.Dropout(p=self.dropout),
                                   nn.Linear(self.hidden_dim // 2, self.vocab_size))

        self.init_s_proj = nn.Linear(self.feat_size, self.hidden_dim)
        self.init_c_proj = nn.Linear(self.feat_size, self.hidden_dim)

        self.baseline_estimator = nn.Linear(self.hidden_dim, 1)

        self.init_weights(0.1)

    def init_weights(self, init_range):
        logging.info("Initialize the parameters of the model")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_range, init_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size, bi, dim):
        # the first parameter from the class
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return Variable(weight.new(self.num_layers * times, batch_size, dim).zero_())
        else:
            return (Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()),
                    Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()))

    def init_hidden_with_feature(self, feature):
        if self.rnn_type == 'gru':
            output = self.init_s_proj(feature)
            return output.view(1, -1, output.size(-1))
        else:
            output1 = self.init_s_proj(feature)
            output2 = self.init_c_proj(feature)
            return (output1.view(1, -1, output1.size(-1)), \
                    output2.view(1, -1, output2.size(-1)))

    def decode(self, imgs, last_word, state_d, penalize_previous=False):
        # 'last_word' is Variable contraining a word index
        # batch_size * input_encoding_size
        word_emb = self.embed(last_word)
        word_emb = torch.unsqueeze(word_emb, 1)

        input_d = torch.cat([word_emb, imgs.unsqueeze(1)], 2)  # batch_size * 1 * dim
        input_d = self.project_d(input_d)

        out_d, state_d = self.decoder(input_d, state_d)

        log_probs = F.log_softmax(self.logit(out_d[:, 0, :]))

        if penalize_previous:
            last_word_onehot = torch.FloatTensor(last_word.size(0), self.vocab_size).zero_().cuda()
            penalize_value = (last_word > 0).data.float() * -100
            mask = Variable(last_word_onehot.scatter_(1, last_word.data[:, None], 1.) * penalize_value[:, None])
            log_probs = log_probs + mask

        return log_probs, state_d

    def forward(self, features, caption):
        """
        :param features: (batch_size, 5, feat_size)
        :param caption: (batch_size, 5, seq_length)
        :return:
        """
        # encode the visual features
        out_e, _ = self.encoder(features)

        # reshape the inputs, making the sentence generation separately
        out_e = out_e.view(-1, out_e.size(2))
        caption = caption.view(-1, caption.size(2))

        ############################# decoding stage ##############################
        batch_size = out_e.size(0)

        # initialize decoder's state
        # state_d = self.init_hidden(batch_size, bi=False, dim=self.hidden_dim)
        state_d = self.init_hidden_with_feature(features)

        last_word = Variable(torch.FloatTensor(batch_size).long().zero_()).cuda()
        outputs = []

        for i in range(self.seq_length):
            log_probs, state_d = self.decode(out_e, last_word, state_d)
            outputs.append(log_probs)

            # choose the word
            if self.ss_prob > 0.0:
                sample_prob = torch.FloatTensor(batch_size).uniform_(0, 1).cuda()
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    last_word = caption[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    last_word = caption[:, i].data.clone()
                    # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(log_probs.data)
                    last_word.index_copy_(0, sample_ind,
                                          torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    last_word = Variable(last_word)
            else:
                last_word = caption[:, i].clone()

            # break condition
            if i >= 1 and caption[:, i].data.sum() == 0:
                break

        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)  # batch_size * 5, -1, vocab_size
        return outputs.view(-1, self.story_size, outputs.size(1), self.vocab_size)

    def sample(self, features, sample_max, rl_training=False, pad=False):
        # encode the visual features
        out_e, _ = self.encoder(features)

        # reshape the inputs, making the sentence generation separately
        out_e = out_e.view(-1, out_e.size(2))

        ###################### Decoding stage ###############################
        batch_size = out_e.size(0)

        # initialize decoder's state
        # state_d = self.init_hidden(batch_size, bi=False, dim=self.hidden_dim)
        state_d = self.init_hidden_with_feature(features)

        seq = []
        seq_log_probs = []
        if rl_training:
            baseline = []

        last_word = torch.FloatTensor(batch_size).long().zero_().cuda()
        for t in range(self.seq_length):
            last_word = Variable(last_word)

            log_probs, state_d = self.decode(out_e, last_word, state_d, True)
            if t < 6:
                mask = np.zeros((batch_size, log_probs.size(-1)), 'float32')
                mask[:, 0] = -1000
                mask = Variable(torch.from_numpy(mask)).cuda()
                log_probs = log_probs + mask

            if sample_max:
                sample_log_prob, last_word = torch.max(log_probs, 1)
                last_word = last_word.data.view(-1).long()
            else:
                # fetch prev distribution: shape Nx(M+1)
                prob_prev = torch.exp(log_probs.data).cpu()
                last_word = torch.multinomial(prob_prev, 1).cuda()
                # gather the logprobs at sampled positions
                sample_log_prob = log_probs.gather(1, Variable(last_word))
                # flatten indices for downstream processing
                last_word = last_word.view(-1).long()

            if t == 0:
                unfinished = last_word > 0
            else:
                unfinished = unfinished * (last_word > 0)
            if unfinished.sum() == 0 and t >= 1 and not pad:
                break
            last_word = last_word * unfinished.type_as(last_word)

            seq.append(last_word)  # seq[t] the input of t time step
            seq_log_probs.append(sample_log_prob.view(-1))

            if rl_training:
                # cut off the gradient passing using detech()
                value = self.baseline_estimator(state_d[0].detach())
                baseline.append(value)

        # concatenate output lists
        seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)  # batch_size * 5, seq_length
        seq_log_probs = torch.cat([_.unsqueeze(1) for _ in seq_log_probs], 1)
        seq = seq.view(-1, self.story_size, seq.size(1))
        seq_log_probs = seq_log_probs.view(-1, self.story_size, seq_log_probs.size(1))

        if rl_training:
            baseline = torch.cat([_.unsqueeze(1) for _ in baseline], 1)  # batch_size * 5, seq_length
            baseline = baseline.view(-1, self.story_size, baseline.size(1))
            return seq, seq_log_probs, baseline
        else:
            return seq, seq_log_probs

    def topK(self, features, beam_size=5):
        assert beam_size <= self.vocab_size and beam_size > 0
        if beam_size == 1:  # if beam_size is 1, then do greedy decoding, otherwise use beam search
            return self.sample(features, sample_max=True, rl_training=False)

        # encode the visual features
        out_e, _ = self.encoder(features)

        # reshape the inputs, making the sentence generation separately
        out_e = out_e.view(-1, out_e.size(2))

        ####################### decoding stage ##################################
        batch_size = out_e.size(0)

        # initialize decoder's state
        state_d = self.init_hidden_with_feature(features)

        topK = []
        for k in range(batch_size):
            out_e_k = out_e[k].unsqueeze(0).expand(beam_size, out_e.size(1)).contiguous()
            state_d_k = state_d[:, k, :].unsqueeze(1).expand(state_d.size(0), beam_size, state_d.size(2)).contiguous()

            last_word = Variable(torch.FloatTensor(beam_size).long().zero_().cuda())  # <BOS>
            log_probs, state_d_k = self.decode(out_e_k, last_word, state_d_k, True)
            log_probs[:, 1] = log_probs[:, 1] - 1000  # never produce <UNK> token
            neg_log_probs = -log_probs

            all_outputs = np.ones((1, beam_size), dtype='int32')
            all_masks = np.ones_like(all_outputs, dtype="float32")
            all_costs = np.zeros_like(all_outputs, dtype="float32")
            for i in range(self.seq_length):
                next_costs = (all_costs[-1, :, None] + neg_log_probs.data.cpu().numpy() * all_masks[-1, :, None])
                (finished,) = np.where(all_masks[-1] == 0)
                next_costs[finished, 1:] = np.inf

                (indexes, outputs), chosen_costs = _smallest(next_costs, beam_size, only_first_row=i == 0)

                new_state_d = state_d_k.data.cpu().numpy()[:, indexes, :]

                all_outputs = all_outputs[:, indexes]
                all_masks = all_masks[:, indexes]
                all_costs = all_costs[:, indexes]

                last_word = Variable(torch.from_numpy(outputs)).cuda()
                state_d_k = Variable(torch.from_numpy(new_state_d)).cuda()

                log_probs, state_d_k = self.decode(out_e_k, last_word, state_d_k, True)

                log_probs[:, 1] = log_probs[:, 1] - 1000
                neg_log_probs = -log_probs

                all_outputs = np.vstack([all_outputs, outputs[None, :]])
                all_costs = np.vstack([all_costs, chosen_costs[None, :]])
                mask = outputs != 0
                all_masks = np.vstack([all_masks, mask[None, :]])
            topK.append(all_outputs[1:].transpose())

        topK = np.asarray(topK, 'int64')
        topK = topK.reshape(-1, 5, topK.shape[1], topK.shape[2])

        return topK

    def predict(self, features, beam_size=5):
        assert beam_size <= self.vocab_size and beam_size > 0
        if beam_size == 1:  # if beam_size is 1, then do greedy decoding, otherwise use beam search
            return self.sample(features, sample_max=True, rl_training=False)

        # encode the visual features
        out_e, _ = self.encoder(features)

        # reshape the inputs, making the sentence generation separately
        out_e = out_e.view(-1, out_e.size(2))

        ####################### decoding stage ##################################
        batch_size = out_e.size(0)

        # initialize decoder's state
        # state_d = self.init_hidden(batch_size, bi=False, dim=self.hidden_dim)
        state_d = self.init_hidden_with_feature(features)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seq_log_probs = torch.FloatTensor(self.seq_length, batch_size)

        # lets process the videos independently for now, for simplicity
        for k in range(batch_size):
            out_e_k = out_e[k].unsqueeze(0).expand(beam_size, out_e.size(1)).contiguous()
            state_d_k = state_d[:, k, :].unsqueeze(1).expand(state_d.size(0), beam_size, state_d.size(2)).contiguous()

            last_word = Variable(torch.FloatTensor(beam_size).long().zero_().cuda())  # <BOS>
            log_probs, state_d_k = self.decode(out_e_k, last_word, state_d_k, True)
            log_probs[:, 1] = log_probs[:, 1] - 1000  # never produce <UNK> token
            neg_log_probs = -log_probs

            all_outputs = np.ones((1, beam_size), dtype='int32')
            all_masks = np.ones_like(all_outputs, dtype="float32")
            all_costs = np.zeros_like(all_outputs, dtype="float32")
            for i in range(self.seq_length):
                if all_masks[-1].sum() == 0:
                    break

                next_costs = (all_costs[-1, :, None] + neg_log_probs.data.cpu().numpy() * all_masks[-1, :, None])
                (finished,) = np.where(all_masks[-1] == 0)
                next_costs[finished, 1:] = np.inf

                (indexes, outputs), chosen_costs = _smallest(next_costs, beam_size, only_first_row=i == 0)

                new_state_d = state_d_k.data.cpu().numpy()[:, indexes, :]

                all_outputs = all_outputs[:, indexes]
                all_masks = all_masks[:, indexes]
                all_costs = all_costs[:, indexes]

                last_word = Variable(torch.from_numpy(outputs)).cuda()
                state_d_k = Variable(torch.from_numpy(new_state_d)).cuda()

                log_probs, state_d_k = self.decode(out_e_k, last_word, state_d_k, True)

                log_probs[:, 1] = log_probs[:, 1] - 1000
                neg_log_probs = -log_probs

                all_outputs = np.vstack([all_outputs, outputs[None, :]])
                all_costs = np.vstack([all_costs, chosen_costs[None, :]])
                mask = outputs != 0
                all_masks = np.vstack([all_masks, mask[None, :]])

            all_outputs = all_outputs[1:]
            all_costs = all_costs[1:] - all_costs[:-1]
            all_masks = all_masks[:-1]
            costs = all_costs.sum(axis=0)
            lengths = all_masks.sum(axis=0)
            normalized_cost = costs / lengths
            best_idx = np.argmin(normalized_cost)
            seq[:all_outputs.shape[0], k] = torch.from_numpy(all_outputs[:, best_idx])
            seq_log_probs[:all_costs.shape[0], k] = torch.from_numpy(all_costs[:, best_idx])

        # return the samples and their log likelihoods
        seq = seq.transpose(0, 1).contiguous()
        seq_log_probs = seq_log_probs.transpose(0, 1).contiguous()
        seq = seq.view(-1, self.story_size, seq.size(1))
        seq_log_probs = seq_log_probs.view(-1, self.story_size, seq_log_probs.size(1))
        return seq, seq_log_probs
