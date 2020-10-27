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
from tqdm import tqdm


def _split_file(params):
    lines = open(params.input_file).readlines()
    cnt = 0
    f_train = open('{0}.train'.format(params.output_base), 'w')
    f_dev = open('{0}.dev'.format(params.output_base), 'w')
    for line in tqdm(lines):
        cnt += 1
        if cnt % 10 == 0:
            f_out = f_dev
        else:
            f_out = f_train
        f_out.write(line)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-file', action='store', dest='input_file',
                      help='Output file')
    parser.add_option('--output-base', action='store', dest='output_base',
                      help='Output base')

    (params, _) = parser.parse_args(sys.argv)

    if params.input_file and params.output_base:
        _split_file(params)
    else:
        parser.print_help()
