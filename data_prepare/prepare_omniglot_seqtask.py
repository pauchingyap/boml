import os
import shutil
import numpy as np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Prepare Omniglot (sequential task)')
    parser.add_argument('--data_path', type=str, help='Parental directory containing all datasets')
    parser.add_argument('--raw_path', type=str, help='Path of folder containing unzipped background and evaluation images')
    parser.add_argument('--dest_dir', type=str, help='Destination directory', default='omniglot_seqtask')
    args = parser.parse_args()

    omni_raw_path = args.raw_path
    dest_path = os.path.join(args.data_path, args.dest_dir)

    # generate omniglot_seqtask folder with split
    train_dir = os.path.join(os.path.join(args.data_path, args.dest_dir), 'metatrain')
    val_dir = os.path.join(os.path.join(args.data_path, args.dest_dir), 'metaval')
    test_dir = os.path.join(os.path.join(args.data_path, args.dest_dir), 'metatest')
    # remove all classes if train, val, test non empty
    if os.path.exists(train_dir):
        print('Train destination folder not empty. Deleting...')
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        print('Val destination folder not empty. Deleting...')
        shutil.rmtree(val_dir)
    if os.path.exists(test_dir):
        print('Test destination folder not empty. Deleting...')
        shutil.rmtree(test_dir)

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)

    train_supercls_ls = np.load('./data_split/omniglot_seqtask/metatrain.npy', allow_pickle=True).tolist()
    val_supercls_ls = np.load('./data_split/omniglot_seqtask/metaval.npy', allow_pickle=True).tolist()
    test_supercls_ls = np.load('./data_split/omniglot_seqtask/metatest.npy', allow_pickle=True).tolist()

    train_supercls_name_ls = [os.path.basename(os.path.normpath(tr_supercls)) for tr_supercls in train_supercls_ls]
    val_supercls_name_ls = [os.path.basename(os.path.normpath(v_supercls)) for v_supercls in val_supercls_ls]
    test_supercls_name_ls = [os.path.basename(os.path.normpath(te_supercls)) for te_supercls in test_supercls_ls]

    # list background and evaluation alphabets
    bg_path = os.path.join(omni_raw_path, 'images_background')
    ev_path = os.path.join(omni_raw_path, 'images_evaluation')
    bg_supercls_ls = os.listdir(bg_path)
    ev_supercls_ls = os.listdir(ev_path)

    train_supercls_dir_ls = [
        os.path.join(bg_path, supercls_name) if supercls_name in bg_supercls_ls
        else os.path.join(ev_path, supercls_name) for supercls_name in train_supercls_name_ls
    ]
    val_supercls_dir_ls = [
        os.path.join(bg_path, supercls_name) if supercls_name in bg_supercls_ls
        else os.path.join(ev_path, supercls_name) for supercls_name in val_supercls_name_ls
    ]
    test_supercls_dir_ls = [
        os.path.join(bg_path, supercls_name) if supercls_name in bg_supercls_ls
        else os.path.join(ev_path, supercls_name) for supercls_name in test_supercls_name_ls
    ]

    for idx, (supercls_dir, supercls_name) in enumerate(zip(train_supercls_dir_ls, train_supercls_name_ls)):
        print('Copying alphabet {} of {}: {}'.format(idx + 1, len(train_supercls_dir_ls), supercls_dir))
        shutil.copytree(supercls_dir, os.path.join(os.path.join(dest_path, 'metatrain'), supercls_name))

    for idx, (supercls_dir, supercls_name) in enumerate(zip(val_supercls_dir_ls, val_supercls_name_ls)):
        print('Copying alphabet {} of {}: {}'.format(idx + 1, len(val_supercls_dir_ls), supercls_dir))
        shutil.copytree(supercls_dir, os.path.join(os.path.join(dest_path, 'metaval'), supercls_name))

    for idx, (supercls_dir, supercls_name) in enumerate(zip(test_supercls_dir_ls, test_supercls_name_ls)):
        print('Copying alphabet {} of {}: {}'.format(idx + 1, len(test_supercls_dir_ls), supercls_dir))
        shutil.copytree(supercls_dir, os.path.join(os.path.join(dest_path, 'metatest'), supercls_name))
