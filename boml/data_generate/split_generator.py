import os
import glob
import cv2


def augment_cls(dir, type='rotation'):
    # only supporting rotation atm
    if type == 'rotation':
        # list all classes in train val test dir
        classlist = glob.glob(dir + '/*')
        len_classlist = len(classlist)

        # augment for train
        for idx, classdir in enumerate(classlist):
            print('Augmenting {}: folder {} of {}'.format(dir, idx + 1, len_classlist))
            # make new folder for rotated 90, 180, 270
            imgdir_rot90 = os.path.normpath(classdir) + '_rotate090'
            imgdir_rot180 = classdir + '_rotate180'
            imgdir_rot270 = classdir + '_rotate270'
            os.makedirs(imgdir_rot90)
            os.makedirs(imgdir_rot180)
            os.makedirs(imgdir_rot270)
            # list all imgs in this dir
            train_imglist = os.listdir(classdir)
            for imgname in train_imglist:
                img = cv2.imread(os.path.join(classdir, imgname))
                (height, width) = img.shape[:-1]
                center = (height / 2, width / 2)

                rotmat90 = cv2.getRotationMatrix2D(center, 90, 1.0)
                rotmat180 = cv2.getRotationMatrix2D(center, 180, 1.0)
                rotmat270 = cv2.getRotationMatrix2D(center, 270, 1.0)

                img_rot90 = cv2.warpAffine(img, rotmat90, (height, width))
                img_rot180 = cv2.warpAffine(img, rotmat180, (width, height))
                img_rot270 = cv2.warpAffine(img, rotmat270, (height, width))

                cv2.imwrite(os.path.join(imgdir_rot90, 'rot090_' + imgname), img_rot90)
                cv2.imwrite(os.path.join(imgdir_rot180, 'rot180_' + imgname), img_rot180)
                cv2.imwrite(os.path.join(imgdir_rot270, 'rot270_' + imgname), img_rot270)
