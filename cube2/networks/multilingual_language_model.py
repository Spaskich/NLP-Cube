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
from cube2.io_utils.conll import ConllEntry
import numpy as np
import optparse
import random
import json
from tqdm import tqdm


class Dataset:
    def __init__(self):
        self.sequences = []

    def load_dataset(self, filename, lang_id):
        lines = open(filename).readlines()
        for line in tqdm(lines):
            parts = line.strip().split(' ')
            self.sequences.append([parts, lang_id])


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

        output_size = np.log(len(encodings.word2int))
        if output_size - int(output_size) > 0:
            output_size += 1
        output_size = int(output_size)
        self._output = nn.Linear(config.proj_size + config.lang_emb_size, output_size)
        self._pad = nn.Embedding(1, config.word_emb_size)

        self._encodigs = encodings
        self._config = config

    def forward(self, sentences, languages, return_out=False):
        x_words, x_langs = self._make_data(sentences, languages)
        x_words = self._word_lookup(x_words)
        x_langs = self._lang_lookup(x_langs)
        x_langs_tmp = x_langs.unsqueeze(1).repeat(1, x_words.shape[1], 1)
        x = torch.cat([x_words, x_langs_tmp], dim=-1)
        x = torch.dropout(x, 0.5, self.training)
        x_fw, _ = self._fw_rnn(x)
        x_bw = torch.flip(self._bw_rnn(torch.flip(x, [1]))[0], [1])
        x_langs = x_langs.unsqueeze(1).repeat(1, x_words.shape[1] - 2, 1)
        x_cat = torch.cat([x_langs, x_fw[:, :-2, :], x_bw[:, 2:, :]], dim=-1)
        x_cat = torch.dropout(x_cat, 0.5, self.training)
        x_proj = torch.tanh(self._proj(x_cat))
        if return_out:
            x_cat2 = torch.cat([x_proj, x_langs], dim=-1)
            x_cat2 = torch.dropout(x_cat2, 0.5, self.training)
            return torch.sigmoid(self._output(x_cat2))
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


def _eval(model, dataset, encodings, criterion, word2bin):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        num_batches = len(dataset.sequences) // params.batch_size
        if len(dataset.sequences) % params.batch_size != 0:
            num_batches += 1

        import tqdm
        pgb = tqdm.tqdm(range(num_batches), desc='\tevaluating loss=N/A', ncols=80)
        for batch_idx in pgb:
            start = batch_idx * params.batch_size
            stop = min(len(dataset.sequences), start + params.batch_size)
            sents, langs = _make_batch(dataset.sequences[start:stop])
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
            total_loss += loss.item()
            pgb.set_description('\tevaluating loss={0:.4f}'.format(loss.item()))

        return total_loss / num_batches


def _to_int(value):
    t = 0
    for ii in range(len(value)):
        t += value[ii] * (2 ** ii)
    return t


def _to_bin(value, max_bin_size):
    d = np.array([value])
    rez = (((d[:, None] & (1 << np.arange(max_bin_size)))) > 0).astype(float)
    return rez[0]


def _prepare_vars(word_list):
    word2int = {}
    word2bin = {}
    word2context = {}
    bin_size = np.log(len(word_list))
    if bin_size - int(bin_size) > 0:
        bin_size += 1
    bin_size = int(bin_size)

    cnt = 0
    for word in word_list:
        word2int[word] = cnt
        word2bin[word] = _to_bin(cnt, bin_size)
        word2context[word] = {'TOTAL': 0, 'ctx': np.zeros((bin_size), dtype=np.float)}
        cnt += 1
    return word2int, word2bin, word2context


def _reorder_words(model, dataset, word2ctx, encodings):
    model.eval()
    with torch.no_grad():
        num_batches = len(dataset.sequences) // params.batch_size
        if len(dataset.sequences) % params.batch_size != 0:
            num_batches += 1

        import tqdm
        pgb = tqdm.tqdm(range(num_batches), desc='\tReordering words', ncols=80)
        for batch_idx in pgb:
            start = batch_idx * params.batch_size
            stop = min(len(dataset.sequences), start + params.batch_size)
            sents, langs = _make_batch(dataset.sequences[start:stop])
            y_pred = model(sents, langs, return_out=True)
            for ii in range(len(sents)):
                for jj in range(len(sents[ii])):
                    word = sents[ii][jj].lower()
                    if word in encodings.word2int:
                        word2ctx[word]['TOTAL'] += 1
                        word2ctx[word]['ctx'] += y_pred[ii, jj, :].detach().cpu().numpy()

                        # y_pred_list.append(y_pred[ii, jj, :].unsqueeze(0))

            # pgb.set_description('\tloss={0:.4f}'.format(loss.item()))
        word2int = {}
        for word in word2ctx:
            if word2ctx[word]['TOTAL'] > 0:
                binary = word2ctx[word]['ctx'] / word2ctx[word]['TOTAL']
            else:
                binary = word2ctx[word]['ctx']
            for ii in range(len(binary)):
                if binary[ii] < 0.5:
                    binary[ii] = 0
                else:
                    binary[ii] = 1
            word2int[word] = _to_int(binary)
        word_list = [k for k, _ in sorted(word2int.items(), key=lambda item: item[1])]
        return word_list


def _get_closest_word(word, unigrams, word_vectors):
    max_simi = 0
    max_word = ''
    from scipy import spatial
    index1 = unigrams[word]
    for w in unigrams:
        if w != word:
            dist = 1 - spatial.distance.cosine(word_vectors[unigrams[w]], word_vectors[index1])
            if dist > max_simi:
                max_simi = dist
                max_word = w
    return max_word, max_simi


def build_word_tree(params):
    trainset = Dataset()
    trainset.load_dataset(params.cluster_train_file, 0)
    max_vocab_size = 30000
    sys.stdout.write('Computing n-gram stats...\n')
    sys.stdout.flush()
    unigram_counts = {}
    total_uni = 0
    for seq in tqdm(trainset.sequences):
        for item in seq[0]:
            token = item.lower()
            if token not in unigram_counts:
                unigram_counts[token] = 1
            else:
                unigram_counts[token] += 1
            total_uni += 1
    # convert to probs
    sorted_data = {k: v for k, v in sorted(unigram_counts.items(), key=lambda item: item[1], reverse=True)}

    tmp = {}
    cnt = 0
    for uni in sorted_data:
        cnt += 1
        tmp[uni] = len(tmp)
        if cnt >= max_vocab_size and max_vocab_size != -1:
            break

    unigrams = tmp
    sys.stdout.write('Unigram count is {0}\n'.format(len(unigrams)))

    sys.stdout.write('Computing cooc patterns...\n')
    sys.stdout.flush()
    cooc = np.zeros((len(unigrams), len(unigrams)))
    for seq in tqdm(trainset.sequences):
        seq = seq[0]
        for ii in range(len(seq)):
            item1 = seq[ii].lower()
            if item1 in unigrams:
                index1 = unigrams[item1]
                # for jj in range(ii, min(len(seq), ii + 5)):
                for jj in range(ii, len(seq)):
                    item2 = seq[jj].lower()
                    if item2 in unigrams:
                        index2 = unigrams[item2]
                        cooc[index1, index2] += 1
                        if index1 != index2:
                            cooc[index2, index1] += 1

    for ii in tqdm(range(cooc.shape[0])):
        cooc[ii] = cooc[ii] / cooc[ii, ii]
        cooc[ii, ii] = 1

    sys.stdout.write('done\n')
    sys.stdout.write('Running SVD...')
    sys.stdout.flush()
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=100, n_iter=15, random_state=1234)
    word_vectors = svd.fit_transform(cooc)
    sys.stdout.write('done\n')

    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=len(unigrams) // 100)
    cluster_labels = clustering.fit_predict(word_vectors)
    cluster2list = {}
    uni_list = ['' for _ in range(len(unigrams))]
    for uni in unigrams:
        uni_list[unigrams[uni]] = uni
    for ii in range(len(cluster_labels)):
        cluster_id = cluster_labels[ii]
        unigram = uni_list[ii]
        if cluster_id not in cluster2list:
            cluster2list[cluster_id] = []
        if unigram[:2] != '##':
            cluster2list[cluster_id].append(unigram)

    f = open(params.cluster_output_file, 'w')
    for cluster_id in cluster2list:
        f.write(str(cluster_id) + '\n')
        f.write(' '.join(cluster2list[cluster_id]) + '\n\n\n')
    f.close()


def do_train(params):
    ds_list = json.load(open(params.train_file))
    train_list = []
    dev_list = []
    for ii in range(len(ds_list)):
        train_list.append(ds_list[ii][1])
        dev_list.append(ds_list[ii][2])

    trainset = Dataset()
    devset = Dataset()
    for ii, train, dev in zip(range(len(train_list)), train_list, dev_list):
        trainset.load_dataset(train_list[ii], ii)
        devset.load_dataset(dev_list[ii], ii)

    encodings = Encodings()
    # encodings.compute(trainset, devset, word_cutoff=2)
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
    word2int, word2bin, word2context = _prepare_vars(word_list)

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
        # if epoch % 20 == 0:
        #     word_list = _reorder_words(model, trainset, word2context, encodings)
        #     f = open('tmp.txt', 'w')
        #     for w in word_list:
        #         f.write(w + '\n')
        #     f.close()
        #     word2int, word2bin, word2context = _prepare_vars(word_list)

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
    parser.add_option('--cluster-train-file', action='store', dest='cluster_train_file',
                      help='Cluster words for h-softmax')
    parser.add_option('--cluster-output-file', action='store', dest='cluster_output_file')
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
    parser.add_option('--max-vocab', action='store', dest='max_vocab_size', default=30000)
    parser.add_option('--stop-words', action='store', dest='stop_words_file')

    (params, _) = parser.parse_args(sys.argv)

    if params.train_file:
        do_train(params)
    elif params.cluster_train_file:
        build_word_tree(params)
