#
# Authors: Tiberiu Boros, Stefan Daniel Dumitrescu
#
# Copyright (c) 2018 Adobe Systems Incorporated. All rights reserved.
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
import ast
from builtins import object, super
from cube.misc.misc import fopen
import collections

if sys.version_info[0] == 2:
    import ConfigParser
else:
    import configparser


class Config(object):
    """Generic base class that implements load/save utilities."""

    def __init__(self):
        """Call to set config object name."""
        self.__config__ = self.__class__.__name__

    def _auto_cast(self, s):
        """Autocasts string s to its original type."""
        try:
            return ast.literal_eval(s)
        except:
            return s

    def save(self, filename):
        """Save configuration to file."""
        sorted_dict = collections.OrderedDict(sorted(self.__dict__.items()))  # sort dictionary
        if sys.version_info[0] == 2:
            config = ConfigParser.ConfigParser()
        else:
            config = configparser.ConfigParser()
        config.add_section(self.__config__)  # write header
        if sys.version_info[0] == 2:
            items = sorted_dict.iteritems()
        else:
            items = sorted_dict.items()
        for k, v in items:  # for python3 use .items()
            if not k.startswith("_"):  # write only non-private properties
                if isinstance(v, float):  # if we are dealing with a float
                    str_v = str(v)
                    if "e" not in str_v and "." not in str_v:  # stop possible confusion with an int by appending a ".0"
                        v = str_v + ".0"
                v = str(v)
                config.set(self.__config__, k, v)
        with fopen(filename, 'w') as cfgfile:
            config.write(cfgfile)

    def load(self, filename):
        """Load configuration from file."""
        if sys.version_info[0] == 2:
            config = ConfigParser.ConfigParser()
        else:
            config = configparser.ConfigParser()
        config.read(filename)
        # check to see if the config file has the appropriate section
        if not config.has_section(self.__config__):
            sys.stderr.write(
                "ERROR: File \"" + filename + "\" is not a valid configuration file for the selected task: Missing section [" + self.__config__ + "]!\n")
            sys.exit(1)
        for k, v in config.items(self.__config__):
            self.__dict__[k] = self._auto_cast(v)


class MLMConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.rnn_layers = 1
        self.rnn_layer_size = 200
        self.word_emb_size = 300
        self.lang_emb_size = 100
        self.proj_size = 300
        self.num_languages = 1

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)

        self._valid = True
