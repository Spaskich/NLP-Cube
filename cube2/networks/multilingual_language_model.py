#
# Author: Tiberiu Boros
#
# Copyright (c) 2020 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import sys

sys.path.append('')
import torch.nn as nn
from cube2.io_utils.encodings import Encodings
from cube2.io_utils.config import MLMConfig
import numpy as np
import optparse
import random
import json
from tqdm import tqdm


class Dataset:
    def __init__(self):
        self.sequences = []
        self.lang2cluster = {}

    def load_dataset(self, filename, lang_id):
        lines = open(filename).readlines()
        for line in tqdm(lines, desc='\t{0}'.format(filename), ncols=100):
            parts = line.strip().split(' ')
            self.sequences.append([parts, lang_id])

    def load_clusters(self, filename, lang_id):
        lines = open(filename).readlines()
        self.lang2cluster[lang_id] = {}
        for ii in tqdm(range(len(lines) // 4), desc='\t{0}'.format(filename), ncols=100):
            self.lang2cluster[lang_id][ii] = lines[ii * 4 + 1].split(' ')


class Encodings:
    def __init__(self, filename=None):
        self._token2int = {}
        self._char2int = {}
        self._num_langs = 0
        self._word2target = {}
        self._max_clusters = 0
        self._max_words_in_clusters = 0
        if filename is not None:
            self.load(filename)

    def save(self, filename):
        json_obj = {'num_langs': self._num_langs, 'max_clusters': self._max_clusters,
                    'max_words_in_clusters': self._max_words_in_clusters, 'token2int': self._token2int,
                    'char2int': self._char2int, '_word2target': self._word2target}
        json.dump(json_obj, open(filename, 'w'))

    def load(self, filename):
        json_obj = json.load(open(filename))
        self._token2int = json_obj['token2int']
        self._char2int = json_obj['char2int']
        self._num_langs = json_obj['num_langs']
        self._word2target = json_obj['word2target']
        self._max_clusters = json_obj['max_clusters']
        self._max_words_in_clusters = json_obj['max_words_in_clusters']

    def compute_encodings(self, dataset: Dataset, w_cutoff=7, ch_cutoff=7):
        char2count = {}
        token2count = {}
        for example in dataset.sequences:
            seq = example[0]
            lang_id = example[1]
            if lang_id + 1 > self._num_langs:
                self._num_langs = lang_id + 1
            for token in seq:
                if token not in token2count:
                    tk = token.lower()
                    token2count[tk] = 1
                else:
                    token2count[tk] += 1
            for char in token:
                ch = char.lower()
                if ch not in char2count:
                    char2count[ch] = 1
                else:
                    char2count[ch] += 1

        self._char2int = {'<PAD>': 0, '<UNK>': 1}
        self._token2int = {'<PAD>': 0, '<UNK>': 1}
        for char in char2count:
            if char2count[char] >= ch_cutoff:
                self._char2int[char] = len(self._char2int)
        for token in token2count:
            if token2count[token] > w_cutoff:
                self._token2int[token] = len(self._token2int)
        self._word2target = {}
        for lang_id in dataset.lang2cluster:
            clusters = dataset.lang2cluster[lang_id]
            self._word2target[lang_id] = {}
            if len(clusters) > self._max_clusters:
                self._max_clusters = len(clusters)
            for cid in clusters:
                cluster = clusters[cid]
                if len(cluster) > self._max_words_in_clusters:
                    self._max_words_in_clusters = len(cluster)
                cnt = 0
                for word in cluster:
                    self._word2target[lang_id][word] = [cid, cnt]
                    cnt += 1

    def __str__(self):
        w2t_count = 0
        for lang_id in self._word2target:
            w2t_count += len(self._word2target[lang_id])
        result = "\t::Holistic tokens: {0}\n\t::Holistic chars: {1}\n\t::Max clusters: {2}\n\t::Max words in clusters: {3}\n\t::Languages: {4}\n\t::Known word targets: {5}".format(
            len(self._token2int), len(self._char2int), self._max_clusters, self._max_words_in_clusters, self._num_langs,
            w2t_count)
        return result


class SkipGram(nn.Module):
    def __init__(self, encodings: Encodings):
        super(SkipGram, self).__init__()
        self._encodings = encodings
        self._lang_emb = nn.Embedding(encodings._num_langs, 32)
        self._tok_emb = nn.Embedding(len(encodings._char2int), 256)
        self._case_emb = nn.Embedding(4, 32)
        self._hmax_emb = nn.Embedding(encodings._max_clusters, 64)
        self._conv = nn.Conv1d(256 + 32 + 32, 256, kernel_size=5)
        self._rnn = nn.LSTM(256 + 32, 256, num_layers=1, batch_first=True)
        self._output_h1 = nn.Linear(200 + 32, encodings._max_clusters)
        self._output_w1 = nn.Linear(200 + 64 + 32, encodings._max_words_in_clusters)
        self._output_h2 = nn.Linear(200 + 32, encodings._max_clusters)
        self._output_w2 = nn.Linear(200 + 64 + 32, encodings._max_words_in_clusters)
        self._output_h3 = nn.Linear(200 + 32, encodings._max_clusters)
        self._output_w3 = nn.Linear(200 + 64 + 32, encodings._max_words_in_clusters)
        self._output_h4 = nn.Linear(200 + 32, encodings._max_clusters)
        self._output_w4 = nn.Linear(200 + 64 + 32, encodings._max_words_in_clusters)

    def forward(self, words, langs, hmax=None):
        x_char, x_case, x_lang, x_hmax = self._make_data(words, langs, hmax)

        x_char = self._tok_emb(x_char)
        x_case = self._case_emb(x_case)
        x_lang = self._lang_emb(x_lang)

        x = torch.cat([x_char, x_case, x_lang], dim=-1)
        conv_out = self._conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        pre_rnn = torch.cat([conv_out, x_lang.unsqueeze(1).repeat(1, x_case.shape[1], 1)], dim=-1)
        rnn_out = self._rnn(pre_rnn)[0][:, -1, :]
        if x_hmax != None:
            x_hmax = self._hmax_emb(x_hmax)

            pre_output_h1 = torch.cat([rnn_out, x_lang], dim=-1)
            output_h1 = self._output1(pre_output_h1)
            pre_output_w1 = torch.cat([rnn_out, x_lang, x_hmax[:, 0, :]], dim=-1)
            output_w1 = self._output2(pre_output_w1)

            pre_output_h2 = torch.cat([rnn_out, x_lang], dim=-1)
            output_h2 = self._output1(pre_output_h2)
            pre_output_w2 = torch.cat([rnn_out, x_lang, x_hmax[:, 1, :]], dim=-1)
            output_w2 = self._output2(pre_output_w2)

            pre_output_h3 = torch.cat([rnn_out, x_lang], dim=-1)
            output_h3 = self._output1(pre_output_h3)
            pre_output_w3 = torch.cat([rnn_out, x_lang, x_hmax[:, 2, :]], dim=-1)
            output_w3 = self._output2(pre_output_w3)

            pre_output_h4 = torch.cat([rnn_out, x_lang], dim=-1)
            output_h4 = self._output1(pre_output_h4)
            pre_output_w4 = torch.cat([rnn_out, x_lang, x_hmax[:, 3, :]], dim=-1)
            output_w4 = self._output2(pre_output_w4)

            return output_h1, output_w1, output_h2, output_w2, output_h3, output_w3, output_h4, output_w4
        else:
            return rnn_out

    def _get_device(self):
        if self._word_lookup.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._word_lookup.weight.device.type, str(self._word_lookup.weight.device.index))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


def _eval(model, dataset, encodings, criterion, word2bin):
    total_loss = 0
    model.eval()
    return 0


def do_train(params):
    ds_list = json.load(open(params.train_file))
    train_list = []
    dev_list = []
    cluster_list = []
    for ii in range(len(ds_list)):
        train_list.append(ds_list[ii][1])
        dev_list.append(ds_list[ii][2])
        cluster_list.append(ds_list[ii][3])

    trainset = Dataset()
    devset = Dataset()
    sys.stdout.write('STEP 1: Loading data\n')
    for ii, train, dev in zip(range(len(train_list)), train_list, dev_list):
        sys.stdout.write('\tLoading language {0}\n'.format(ii))
        trainset.load_dataset(train_list[ii], ii)
        devset.load_dataset(dev_list[ii], ii)
        trainset.load_clusters(cluster_list[ii], ii)

    sys.stdout.write('STEP 2: Computing encodings\n')
    encodings = Encodings()
    encodings.compute_encodings(trainset)
    print(encodings)
    sys.exit(0)
    config = MLMConfig()
    config.num_languages = len(train_list)
    if params.config_file:
        config.load(params.config_file)
    # model = MLM(encodings, config)
    model = None
    # if params.device != 'cpu':
    #     model.to(params.device)

    import torch.optim as optim
    import torch.nn as nn
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)  # , betas=(0.9, 0.9))
    optimizer = None
    criterion = nn.CrossEntropyLoss()

    patience_left = params.patience
    epoch = 1

    best_nll = 9999
    encodings.save('{0}.encodings'.format(params.store))
    model._config.save('{0}.conf'.format(params.store))
    word_list = [word for word in encodings.word2int]

    while patience_left > 0:
        patience_left -= 1
        sys.stdout.write('\n\nStarting epoch ' + str(epoch) + '\n')
        sys.stdout.flush()
        random.shuffle(trainset.sequences)
        num_batches = len(trainset.sequences) // params.batch_size
        if len(trainset.sequences) % params.batch_size != 0:
            num_batches += 1
        total_words = 0
        epoch_loss = 0
        import tqdm
        pgb = tqdm.tqdm(range(num_batches), desc='\tloss=NaN', ncols=80)
        model.train()
        for batch_idx in pgb:
            start = batch_idx * params.batch_size
            stop = min(len(trainset.sequences), start + params.batch_size)
            sents, langs = _make_batch(trainset.sequences[start:stop])
            y_pred = model(sents, langs, return_out=True)
            y_tar = []
            y_pred_list = []
            for ii in range(len(sents)):
                for jj in range(len(sents[ii])):
                    word = sents[ii][jj].lower()
                    if word in encodings.word2int:
                        y_tar.append(word2bin[word])
                        y_pred_list.append(y_pred[ii, jj, :].unsqueeze(0))
            y_tar = torch.tensor(y_tar, device=params.device, dtype=torch.float)
            y_pred = torch.cat(y_pred_list, dim=0)
            loss = criterion(y_pred, y_tar)
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            pgb.set_description('\tloss={0:.4f}'.format(loss.item()))

        nll = _eval(model, devset, encodings, criterion, word2bin)

        sys.stdout.write('\tStoring {0}.last\n'.format(params.store))
        sys.stdout.flush()
        fn = '{0}.last'.format(params.store)
        model.save(fn)
        sys.stderr.flush()
        if best_nll > nll:
            best_nll = nll
            sys.stdout.write('\tStoring {0}.bestNLL\n'.format(params.store))
            sys.stdout.flush()
            fn = '{0}.bestNLL'.format(params.store)
            model.save(fn)
            patience_left = params.patience

        sys.stdout.write("\tAVG Epoch loss = {0:.6f}\n".format(epoch_loss / num_batches))
        sys.stdout.flush()
        sys.stdout.write(
            "\tValidation NLL={0:.4f}\n".format(nll))
        sys.stdout.flush()
        epoch += 1


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', action='store', dest='train_file',
                      help='Start building a tagger model')
    parser.add_option('--config', action='store', dest='config_file',
                      help='Use this configuration file for tagger')
    parser.add_option('--patience', action='store', type='int', default=20, dest='patience',
                      help='Number of epochs before early stopping (default=20)')
    parser.add_option('--store', action='store', dest='store', help='Output base', default='mlm')
    parser.add_option('--batch-size', action='store', type='int', default=32, dest='batch_size',
                      help='Number of epochs before early stopping (default=32)')
    parser.add_option('--device', action='store', dest='device', default='cpu',
                      help='What device to use for models: cpu, cuda:0, cuda:1 ...')

    (params, _) = parser.parse_args(sys.argv)

    if params.train_file:
        do_train(params)
