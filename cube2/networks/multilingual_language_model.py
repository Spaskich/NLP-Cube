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


class MLM(nn.Module):
    def __init__(self, encodings: Encodings, config: MLMConfig):
        super(MLM, self).__init__()
        # _start and _end encodings
        self._encodings = encodings
        self._start = len(self._encodings.word2int)
        self._end = len(self._encodings.word2int) + 1
        self._unk = self._encodings.word2int['<UNK>']

        self._word_lookup = nn.Embedding(len(encodings.word2int) + 2, config.word_emb_size)
        self._lang_lookup = nn.Embedding(config.num_languages, config.lang_emb_size)
        self._fw_rnn = nn.LSTM(config.word_emb_size + config.lang_emb_size, config.rnn_layer_size,
                               num_layers=config.rnn_layers, batch_first=True)

        self._bw_rnn = nn.LSTM(config.word_emb_size + config.lang_emb_size, config.rnn_layer_size,
                               num_layers=config.rnn_layers, batch_first=True)

        self._proj = nn.Linear(config.rnn_layer_size * 2 + config.lang_emb_size, config.proj_size)

        self._output = nn.Linear(config.proj_size + config.lang_emb_size, len(encodings.word2int))
        self._pad = nn.Embedding(1, config.word_emb_size)

        self._encodigs = encodings
        self._config = config

    def forward(self, sentences, languages):
        x_words, x_langs = self._make_data(sentences, languages)
        x_words = self._word_lookup(x_words)
        x_langs = self._lang_lookup(x_langs)
        x_langs_tmp = x_langs.unsqueeze(1).repeat(1, x_words.shape[1], 1)
        x = torch.cat([x_words, x_langs_tmp], dim=-1)
        x_fw, _ = self._fw_rnn(x)
        x_bw = torch.flip(self._bw_rnn(torch.flip(x, [1]))[0], [1])
        x_langs = x_langs.unsqueeze(1).repeat(1, x_words.shape[1] - 2, 1)
        x_cat = torch.cat([x_langs, x_fw[:, :-2, :], x_bw[:, 2:, :]], dim=-1)
        x_proj = self._proj(x_cat)
        if self.training:
            x_cat2 = torch.cat([x_proj, x_langs], dim=-1)
            return self._output(x_cat2)
        else:
            return x_proj

    def _make_data(self, sents, langs):
        x_langs = torch.tensor(langs, device=self._get_device(), dtype=torch.long)
        max_words = max([len(sent) for sent in sents])
        x_words = np.zeros((len(sents), max_words + 2), dtype=np.long)
        for ii in range(len(sents)):
            x_words[ii, 0] = self._start
            for jj in range(len(sents[ii])):
                word = sents[ii][jj].lower()
                if word in self._encodigs.word2int:
                    x_words[ii, jj + 1] = self._encodigs.word2int[word]
                else:
                    x_words[ii, jj + 1] = self._unk
            x_words[ii, len(sents[ii]) + 1] = self._end
        x_words = torch.tensor(x_words, device=self._get_device(), dtype=torch.long)
        return x_words, x_langs

    def _get_device(self):
        if self._word_lookup.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._word_lookup.weight.device.type, str(self._word_lookup.weight.device.index))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


def _make_batch(examples):
    sents = []
    langs = []
    for example in examples:
        langs.append(example[1])
        sent = []
        for token in example[0]:
            sent.append(token.word)
        sents.append(sent)
    return sents, langs


def _eval(model, dataset, encodings):
    return 0


def _start_train(params, trainset, devset, encodings, model, criterion, optimizer):
    patience_left = params.patience
    epoch = 1

    best_nll = 0
    encodings.save('{0}.encodings'.format(params.store))
    model._config.save('{0}.conf'.format(params.store))
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
            y_pred = model(sents, langs)
            y_tar = []
            y_pred_list = []
            for ii in range(len(sents)):
                for jj in range(len(sents[ii])):
                    word = sents[ii][jj].lower()
                    if word in encodings.word2int:
                        y_tar.append(encodings.word2int[word])
                        y_pred_list.append(y_pred[ii, jj, :].unsqueeze(0))
            y_tar = torch.tensor(y_tar, device=params.device, dtype=torch.long)
            y_pred = torch.cat(y_pred_list, dim=0)
            loss = criterion(y_pred, y_tar)
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            pgb.set_description('\tloss={0:.4f}'.format(loss.item()))

        nll = _eval(model, devset, encodings)
        sys.stdout.write('\tStoring {0}.last\n'.format(params.store))
        sys.stdout.flush()
        fn = '{0}.last'.format(params.store)
        model.save(fn)
        sys.stderr.flush()
        if best_nll < nll:
            best_nll = nll
            sys.stdout.write('\tStoring {0}.bestNLL\n'.format(params.store))
            sys.stdout.flush()
            fn = '{0}.bestNLL'.format(params.store)
            model.save(fn)
            patience_left = params.patience

        sys.stdout.write("\tAVG Epoch loss = {0:.6f}\n".format(epoch_loss / num_batches))
        sys.stdout.flush()
        sys.stdout.write(
            "\tValidation accuracy NLL={0:.4f}\n".format(nll))
        sys.stdout.flush()
        epoch += 1


def do_train(params):
    from cube2.io_utils.conll import Dataset
    from cube2.io_utils.encodings import Encodings
    from cube2.io_utils.config import MLMConfig
    import json
    ds_list = json.load(open(params.train_file))
    train_list = []
    dev_list = []
    for ii in range(len(ds_list)):
        train_list.append(ds_list[ii][1])
        dev_list.append(ds_list[ii][2])

    trainset = Dataset()
    devset = Dataset()
    for ii, train, dev in zip(range(len(train_list)), train_list, dev_list):
        trainset.load_language(train, ii)
        devset.load_language(dev, ii)
    encodings = Encodings()
    encodings.compute(trainset, devset, word_cutoff=2)
    config = MLMConfig()
    config.num_languages = len(train_list)
    if params.config_file:
        config.load(params.config_file)
    model = MLM(encodings, config)
    if params.device != 'cpu':
        model.to(params.device)

    import torch.optim as optim
    import torch.nn as nn
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # , betas=(0.9, 0.9))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if params.device != 'cpu':
        criterion.cuda(params.device)
    _start_train(params, trainset, devset, encodings, model, criterion, optimizer)


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
