from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from six.moves import cPickle
from collections import defaultdict
from textblob import TextBlob


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1, n + 1):
        for i in xrange(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  # lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.iteritems()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def build_dict(params):
    story_line = json.load(open(params['input_json'], 'r'))
    wtoi = story_line['words2id']

    count = 0
    refs_words = []
    for stories in story_line['album2stories'][params['split']].values():
        ref_words = []
        for story_id in stories:
            txt = story_line[params['split']][story_id]['origin_text']
            tmp_tokens = TextBlob(txt).tokens + ['<EOS>']
            tmp_tokens = [_ if _ in wtoi else '<UNK>' for _ in tmp_tokens]
            ref_words.append(' '.join(tmp_tokens))
        refs_words.append(ref_words)
        count += 1
    print('total albums: ', count)

    ngram_words = compute_doc_freq(create_crefs(refs_words))

    return ngram_words, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='VIST/story_line.json',
                        help='the input json file from the dataset')
    parser.add_argument('--output_pkl', default='VIST/VIST-train', help='output pickle file')
    parser.add_argument('--split', default='train', help='test, val, train')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    ngram_words, ref_len = build_dict(params)

    cPickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl'] + '-words.p', 'w'),
                 protocol=cPickle.HIGHEST_PROTOCOL)
