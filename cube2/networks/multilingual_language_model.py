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


class MLM(nn.Module):
    def __init__(self, encodings: Encodings, config: MLMConfig):
        super(MLM, self).__init__()
        # _start and _end encodings
        self._start = len(self._encodigs.word2int)
        self._end = len(self._encodigs.word2int) + 1

        self._word_lookup = nn.Embedding(len(encodings.word2int) + 2, config.word_emb_size)
        self._lang_lookup = nn.Embedding(config.num_languages, config.lang_emb_size)
        self._fw_rnn = nn.LSTM(config.word_emb_size + config.lang_emb_size, config.rnn_layer_size,
                               num_layers=config.rnn_layers, batch_first=True)

        self._bw_rnn = nn.LSTM(config.word_emb_size + config.lang_emb_size, config.rnn_layer_size,
                               num_layers=config.rnn_layers, batch_first=True)

        self._proj = nn.Linear(config.rnn_layer_size * 2 + config.lang_emb_size, config.proj_size)

        self._output = nn.Linear(config.proj_size, len(encodings.word2int))
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
        x_bw, _ = torch.flip(self._bw_rnn(torch.flip(x, [1])), [1])
        x_langs = x_langs.unsqueeze(1).repeat(1, x_words.shape[1] - 2)
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
                    sents[ii, jj + 1] = self._encodigs.word2int[word]
            x_words[ii, len(sents[ii]) + 1] = self._end
        x_words = torch.tensor(x_words, device=self._get_device(), dtype=torch.long)
        return x_words, x_langs

    def _get_device(self):
        if self.case_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self.case_emb.weight.device.type, str(self.case_emb.weight.device.index))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
