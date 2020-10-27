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

import sys

sys.path.append('')
import numpy as np
import optparse
from tqdm import tqdm


class Dataset:
    def __init__(self):
        self.sequences = []

    def load_dataset(self, filename, lang_id):
        lines = open(filename).readlines()
        for line in tqdm(lines):
            parts = line.strip().split(' ')
            self.sequences.append([parts, lang_id])


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
        cooc[ii, ii] = 0
        cooc[ii] = cooc[ii] / np.max(cooc[ii])

    sys.stdout.write('done\n')
    sys.stdout.write('Running dim reduction...\n')
    sys.stdout.flush()
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=100, n_iter=100, random_state=1234)
    word_vectors = svd.fit_transform(cooc)

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
        cluster2list[cluster_id].append(unigram)

    f = open(params.cluster_output_file, 'w')
    for cluster_id in cluster2list:
        f.write(str(cluster_id) + '\n')
        f.write(' '.join(cluster2list[cluster_id]) + '\n\n\n')
    f.close()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--cluster-train-file', action='store', dest='cluster_train_file',
                      help='Cluster words for h-softmax')
    parser.add_option('--cluster-output-file', action='store', dest='cluster_output_file')
    parser.add_option('--max-vocab', action='store', dest='max_vocab_size', default=30000)

    (params, _) = parser.parse_args(sys.argv)

    if params.cluster_train_file:
        build_word_tree(params)
