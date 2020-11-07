import optparse
import sys
import os
import bz2

sys.path.append('')


def _get_all_files(folder):
    all_files = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if name[-4:] == '.bz2':
                all_files.append(os.path.join(path, name))

    return all_files


def extract(params):
    sys.stdout.write('Scanning input folder\n')
    all_files = _get_all_files(params.source_folder)
    o_file = open(params.output_file, 'w')
    for file in all_files:
        sys.stdout.write('Reading {0}\n'.format(file))
        f = bz2.open(file, mode='rt')
        for line in f.readlines():
            if line[:4] != '<doc' and line[:5] != '</doc' and len(line.strip()) > params.min_chars:
                o_file.write(line.strip() + '\n')

        f.close()
    o_file.close()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--source-folder', action='store', dest='source_folder',
                      help='Location of wikiextractor files')
    parser.add_option('--output-file', action='store', dest='output_file',
                      help='File to output all wiki content')
    parser.add_option('--min-chars', action='store', type='int', default=50, dest='min_chars',
                      help='Minimum number of characters in a sentence (default=50)')

    (params, _) = parser.parse_args(sys.argv)

    if params.source_folder and params.output_file:
        extract(params)
