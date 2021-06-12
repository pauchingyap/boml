import numpy as np
import glob
import os
from tqdm import tqdm
from random import sample
import torchvision.transforms as transforms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Prepare mini-QuickDraw')
    parser.add_argument('--data_path', type=str, help='Parental directory containing all datasets')
    parser.add_argument('--npy_path', type=str, help='Path of folder containing all quickdraw .npy')
    parser.add_argument('--dest_dir', type=str, help='Destination directory', default='mini_quickdraw')
    args = parser.parse_args()

    num_ch = 3
    num_cls_to_sample = None
    num_inst_per_cls_to_sample = 1000

    full_cls_dir_ls = glob.glob(args.npy_path + '/*')
    if num_cls_to_sample is None:
        num_cls_to_sample = len(full_cls_dir_ls)
    mini_quickdraw_cls_dir_ls = sample(full_cls_dir_ls, num_cls_to_sample)

    for npyfile in tqdm(mini_quickdraw_cls_dir_ls, desc='Generating mini_quickdraw'):
        # each npy file is a class (nsample in this class x 784)
        cls_name = os.path.basename(os.path.splitext(os.path.normpath(npyfile))[0])
        npy_cls = np.load(npyfile)
        if num_inst_per_cls_to_sample is None:
            num_inst_per_cls_to_sample = npy_cls.shape[0]
        for idx, imgvec in enumerate(list(npy_cls[sample(range(npy_cls.shape[0]), num_inst_per_cls_to_sample), :])):
            # numpy 2-d image
            img_np = imgvec.reshape(28, 28)
            # masks
            if num_ch == 3:
                bg_mask = np.repeat(np.expand_dims((img_np == 0), axis=0), repeats=3, axis=0)
                obj_mask = np.invert(bg_mask)
                # expand to 3 channels
                img_np = np.repeat(np.expand_dims(img_np, axis=0), repeats=3, axis=0)
            elif num_ch == 1:
                bg_mask = img_np == 0
                obj_mask = np.invert(bg_mask)
            else:
                raise ValueError('num_ch either 1 or 3')
            np.place(img_np, mask=bg_mask, vals=255)
            np.place(img_np, mask=obj_mask, vals=0)
            # transform to pil image
            pil_img = transforms.ToPILImage()(np.transpose(img_np, (1, 2, 0)))
            img_dir = os.path.join(os.path.join(args.data_path, args.dest_dir), cls_name)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            img_name = '{}_{}'.format(cls_name, str(idx))
            pil_img.save(os.path.join(img_dir, img_name), format='PNG')
