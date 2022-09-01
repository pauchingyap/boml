import os
import numpy as np
import shutil

from boml.data_generate.split_generator import augment_cls


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Prepare Omniglot')
    parser.add_argument('--data_path', type=str, help='Parental directory containing all datasets')
    parser.add_argument('--raw_path', type=str, help='Path of folder containing unzipped background and evaluation images')
    parser.add_argument('--dest_dir', type=str, help='Destination directory', default='omniglot')
    args = parser.parse_args()

    omni_raw_path = args.raw_path
    dest_dir = os.path.join(args.data_path, args.dest_dir)

    # copy from raw omniglot to folder 'omniglot' based on split lists provided
    metatrain_full = np.load('./data_split/omniglot/metatrain.npy', allow_pickle=True).tolist()
    metaval_full = np.load('./data_split/omniglot/metaval.npy', allow_pickle=True).tolist()
    metatest_full = np.load('./data_split/omniglot/metatest.npy', allow_pickle=True).tolist()

    # take away rotation information from all paths
    metatrain_chars_long = [
        path.replace('_rotate090', '') if '_rotate090' in path
        else path.replace('_rotate180', '') if '_rotate180' in path
        else path.replace('_rotate270', '') for path in metatrain_full
    ]
    metaval_chars_long = [
        path.replace('_rotate090', '') if '_rotate090' in path
        else path.replace('_rotate180', '') if '_rotate180' in path
        else path.replace('_rotate270', '') for path in metaval_full
    ]
    metatest_chars_long = [
        path.replace('_rotate090', '') if '_rotate090' in path
        else path.replace('_rotate180', '') if '_rotate180' in path
        else path.replace('_rotate270', '') for path in metatest_full
    ]
    metatrain_chars = list(dict.fromkeys(metatrain_chars_long))
    metaval_chars = list(dict.fromkeys(metaval_chars_long))
    metatest_chars = list(dict.fromkeys(metatest_chars_long))

    # get the character paths by detaching basename and replacing _char with /char
    metatrain_char_dirs = [os.path.basename(os.path.normpath(path)).replace('_character', '/character') for path in metatrain_chars]
    metaval_char_dirs = [os.path.basename(os.path.normpath(path)).replace('_character', '/character') for path in metaval_chars]
    metatest_char_dirs = [os.path.basename(os.path.normpath(path)).replace('_character', '/character') for path in metatest_chars]

    metatrain_alph_char = [os.path.split(metatrain_char_dir) for metatrain_char_dir in metatrain_char_dirs]
    metaval_alph_char = [os.path.split(metaval_char_dir) for metaval_char_dir in metaval_char_dirs]
    metatest_alph_char = [os.path.split(metatest_char_dir) for metatest_char_dir in metatest_char_dirs]

    # list background and evaluation alphabets
    bg_path = os.path.join(omni_raw_path, 'images_background')
    ev_path = os.path.join(omni_raw_path, 'images_evaluation')
    bg_supercls_ls = os.listdir(bg_path)
    ev_supercls_ls = os.listdir(ev_path)

    # create new omniglot set
    metatrain_dest_dir = os.path.join(dest_dir, 'metatrain')
    metaval_dest_dir = os.path.join(dest_dir, 'metaval')
    metatest_dest_dir = os.path.join(dest_dir, 'metatest')
    if not os.path.exists(dest_dir):
        os.makedirs(metatrain_dest_dir)
        os.makedirs(metaval_dest_dir)
        os.makedirs(metatest_dest_dir)

    for i, dir in enumerate(metatrain_char_dirs):
        alphabet, character = os.path.split(dir)
        char_path = os.path.join(bg_path, dir) if alphabet in bg_supercls_ls else os.path.join(ev_path, dir)
        print('Copying metatrain character {} of {}: {}'.format(i + 1, len(metatrain_char_dirs), dir))
        shutil.copytree(char_path, os.path.join(metatrain_dest_dir, '{}_{}'.format(alphabet, character)))

    for i, dir in enumerate(metaval_char_dirs):
        alphabet, character = os.path.split(dir)
        char_path = os.path.join(bg_path, dir) if alphabet in bg_supercls_ls else os.path.join(ev_path, dir)
        print('Copying metaval character {} of {}: {}'.format(i + 1, len(metaval_char_dirs), dir))
        shutil.copytree(char_path, os.path.join(metaval_dest_dir, '{}_{}'.format(alphabet, character)))

    for i, dir in enumerate(metatest_char_dirs):
        alphabet, character = os.path.split(dir)
        char_path = os.path.join(bg_path, dir) if alphabet in bg_supercls_ls else os.path.join(ev_path, dir)
        print('Copying metatest character {} of {}: {}'.format(i + 1, len(metatest_char_dirs), dir))
        shutil.copytree(char_path, os.path.join(metatest_dest_dir, '{}_{}'.format(alphabet, character)))

    augment_cls(dir=metatrain_dest_dir, type='rotation')
    augment_cls(dir=metaval_dest_dir, type='rotation')
    augment_cls(dir=metatest_dest_dir, type='rotation')
