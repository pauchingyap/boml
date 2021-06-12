import os
from PIL import Image
from tqdm import tqdm


def prepare_images_and_classes(image_folder_dir, label_only_file_dir, image_label_file_dir_list, crop_bottom_pixel=20):
    # create class folders
    lbl_txt_file = open(label_only_file_dir)
    label_list = lbl_txt_file.read().split('\n')[:-1]
    for lbl in label_list:
        if '/' in lbl:
            lbl = lbl.replace('/', '-', 1)
        os.makedirs(os.path.join(image_folder_dir, lbl))

    # each str in list is of the form: 'image label'
    image_label_list = []
    for image_label_file_dir in image_label_file_dir_list:
        img_lbl_txt_file = open(image_label_file_dir)
        image_label_list += img_lbl_txt_file.read().split('\n')[:-1]

    # crop away bottom copyright info and move to class folders
    for img_lbl in tqdm(image_label_list, desc='Preparing images'):
        img, lbl = img_lbl.split(sep=' ', maxsplit=1)
        img_file_name = img + '.jpg'

        if '/' in lbl:
            lbl = lbl.replace('/', '-', 1)

        ori_img_dir = os.path.join(image_folder_dir, img_file_name)
        dest_img_dir = os.path.join(os.path.join(image_folder_dir, lbl), img_file_name)

        # crop image before moving to class folder
        im = Image.open(ori_img_dir)
        im_width, im_height = im.size
        im_crop = im.crop((0, 0, im_width, im_height - crop_bottom_pixel))
        # save cropped image in class folder
        im_crop.save(fp=dest_img_dir)

        # remove original non-cropped image file
        os.remove(ori_img_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Prepare Aircraft')
    parser.add_argument('--data_path', type=str, help='Parental directory containing all datasets')
    parser.add_argument('--label_path', type=str, help='Path of folder containing "images_variant_test.txt", "images_variant_trainval.txt", "variants.txt"')
    parser.add_argument('--dest_dir', type=str, help='Destination directory', default='aircraft')
    args = parser.parse_args()

    dest_path = os.path.join(args.data_path, args.dest_dir)

    # organise images into class folders
    prepare_images_and_classes(
        image_folder_dir=dest_path,
        label_only_file_dir=os.path.join(args.label_path, 'variants.txt'),
        image_label_file_dir_list = [os.path.join(args.label_path, 'images_variant_trainval.txt'),
                                     os.path.join(args.label_path, 'images_variant_test.txt')],
        crop_bottom_pixel=20
    )
