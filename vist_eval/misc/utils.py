import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def decode_sequence(ix_to_word, seq):
    '''
    Input: seq is a tensor of size (batch_size, seq_length), with element 0 .. vocab_size. 0 is <END> token.
    '''
    if isinstance(seq, list):
        out = []
        for i in range(len(seq)):
            txt = ''
            for j in range(len(seq[i])):
                ix = seq[i][j]
                if ix > 0:
                    if j >= 1:
                        txt = txt + ' '
                    txt = txt + ix_to_word[str(ix)]
                else:
                    break
            out.append(txt)
        return out
    else:
        N, D = seq.size()
        out = []
        for i in range(N):
            txt = ''
            for j in range(D):
                ix = seq[i, j]
                if ix > 0:
                    if j >= 1:
                        txt = txt + ' '
                    txt = txt + ix_to_word[str(ix)]
                else:
                    break
            out.append(txt)
        return out


def decode_story(id2word, result):
    """
    :param id2word: vocab
    :param result: (batch_size, story_size, seq_length)
    :return:
    out: a list of stories. the size of the list is batch_size
    """
    batch_size, story_size, seq_length = result.size()
    out = []
    for i in xrange(batch_size):
        txt = ''
        for j in xrange(story_size):
            for k in xrange(seq_length):
                vocab_id = result[i, j, k]
                if vocab_id > 0:
                    txt = txt + ' ' + id2word[str(vocab_id)]
                else:
                    break
        out.append(txt)
    return out

def post_process_story(id2word, result):
    """
    :param id2word: vocab
    :param result: (batch_size, beam_size, story_size, seq_length)
    :return:
    out: a list of stories. the size of the list is batch_size
    """
    batch_size, story_size, beam_size, seq_length = result.shape
    out = []
    for i in xrange(batch_size):
        txts = []
        stories = []
        for j in xrange(story_size):
            for b in xrange(beam_size):
                txt = ''
                for k in xrange(seq_length):
                    vocab_id = result[i, j, b, k]
                    if vocab_id > 0:
                        txt = txt + ' ' + id2word[str(vocab_id)]
                    else:
                        break
            stories.append(txt)
        txts.append(stories)
        out.append(txts)
    return out


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
