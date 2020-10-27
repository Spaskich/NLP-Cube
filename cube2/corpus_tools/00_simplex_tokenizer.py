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
import optparse
import re
import unicodedata


def normalize(word):
    return unicodedata.normalize('NFD', word).encode('ascii', 'ignore')


def is_stopword(stopwords, word):
    word = normalize(word.lower())
    return word in stopwords


def _tokenize_file(params):
    if params.stopwords_file:
        stopwords = {}
        for line in open(params.stopwords_file).readlines():
            stopwords[normalize(line.strip())] = 1
    else:
        stopwords = None
    f_out = open(params.output_file, 'w')
    f_in = open(params.input_file)
    lines = f_in.readlines()
    from tqdm import tqdm
    for line in tqdm(lines):
        if line.strip() != '':
            new_line = re.sub(r'[^\w0-9 ]+', ' ', line)
            nn = new_line.replace('  ', ' ')
            while nn != new_line:
                new_line = nn
                nn = new_line.replace('  ', ' ')
            new_line = re.sub('\d', '0', new_line)
            if stopwords is not None:
                parts = new_line.split(' ')
                new_pp = []
                for part in parts:
                    if not is_stopword(stopwords, part):
                        new_pp.append(part)
                new_line = ' '.join(new_pp)
            f_out.write(new_line + '\n')
    f_out.close()
    f_in.close()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-file', action='store', dest='input_file',
                      help='Output file')
    parser.add_option('--output-file', action='store', dest='output_file',
                      help='Output file')
    parser.add_option('--stop-words', action='store', dest='stopwords_file')

    (params, _) = parser.parse_args(sys.argv)

    if params.input_file and params.output_file:
        _tokenize_file(params)
    else:
        parser.print_help()
