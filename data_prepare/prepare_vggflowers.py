import os
from scipy.io import loadmat


def organise_class_folders(image_folder_dir, label_mat_dir):
    image_file_ls = [
        'image_{}.jpg'.format('0' * (5 - len(str(i))) + str(i)) for i in range(1, len(os.listdir(image_folder_dir)) + 1)
    ]
    label_ls = list(loadmat(label_mat_dir)['labels'][0])
    image_label_dict = dict(zip(image_file_ls, label_ls))

    # create class folders
    for label_int in list(set(label_ls)):
        os.makedirs(os.path.join(image_folder_dir, str(label_int)))

    # move files into respective folder
    for image, label in image_label_dict.items():
        os.rename(
            src=os.path.join(image_folder_dir, image),
            dst=os.path.join(os.path.join(image_folder_dir, str(label)), image)
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Organise VGG-Flowers into classes')
    parser.add_argument('--data_path', type=str, help='Parental directory containing all datasets', default='../data')
    parser.add_argument('--image_folder', type=str, help='VGG-Flowers folder name', default='vggflowers')
    parser.add_argument('--label_path', type=str, help='Path of .mat label', )
    args = parser.parse_args()

    # organise images into class folders
    organise_class_folders(
        image_folder_dir=os.path.join(args.data_path, args.image_folder),
        label_mat_dir=args.label_path
    )
