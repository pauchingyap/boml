## Data preparation
We pre-process the datasets and put them into a desired parental data folder `$DATA_PATH` as below.


### Aircraft

1. Download "Data, annotations, and evaluation" [here](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).

1. Extract the files and put all the jpg images into the folder `aircraft` under `$DATA_PATH`:
   ```bash
   $DATA_PATH
   └── aircraft
       ├── 0034309.jpg
       ├── 0034958.jpg
       ├── ...
   ```

2. Run `data_prepare/prepare_aircraft.py` with options:
   + `--data_path`: `$DATA_PATH` as above
   + `--label_path`: Path of folder containing "images_variant_test.txt", "images_variant_trainval.txt", "variants.txt" (eg: `~/fgvc-aircraft-2013b/data/`)


### CIFAR-FS and *mini*ImageNet
Download the CIFAR-FS and *mini*ImageNet datasets [here](https://github.com/bertinetto/r2d2), and extract the images into folder `cifar100` and `mini_imagenet` in `$DATA_PATH`:
```bash
$DATA_PATH
├── cifar100
│   ├── apple
│   │   ├── ...
│   ├── aquarium_fish
│   │   ├── ...
│   ├── ...
└── mini_imagenet
    ├── n01532829
    │   ├── ...
    ├── n01558993
    │   ├── ...
    ├── ...
```


### *mini*QuickDraw

1. Download the `.npy` files of all QuickDraw classes [here](https://github.com/googlecreativelab/quickdraw-dataset), and put them in a folder (eg: `quickdraw_npy`):
   ```bash
   $DATA_PATH
   └── quickdraw_npy
       ├── aircraft carrier.npy
       ├── airplane.npy
       ├── ...
   ```

1. Run `data_prepare/prepare_quickdraw.py` with options:
   + `--data_path`: `$DATA_PATH` as above
   + `--npy_path`: path of the folder containing all downloaded `.npy` files (eg: `~/$DATA_PATH/quickdraw_npy`)


### Omniglot

1. Download the python "images_background.zip" and "images_evaluation.zip" [here](https://github.com/brendenlake/omniglot) and extract them into a folder `$RAW_OMNIGLOT_PATH`:
   ```bash
   $RAW_OMNIGLOT_PATH
   ├── images_background
   │   ├── Alphabet_of_the_Magi
   │   ├── ... 
   └── images_evaluation
       ├── Angelic
       ├── ... 
   ```

1. Run `data_prepare/prepare_omniglot.py` and `data_prepare/prepare_omniglot_seqtask.py` with options:

   + `--data_path`: `$DATA_PATH` as above
   + `--raw_path`: `$RAW_OMNIGLOT_PATH` as above


### VGG-Flowers

1. Download dataset images and image labels [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

1. Extract the images to the parental data directory and rename the folder of images to `vggflowers`. The directory structure should look like:
    ```bash
    $DATA_PATH
    └── vggflowers
        ├── image_00001.jpg
        ├── image_00002.jpg
        ├── ...
    ```
1. Run `data_prepare/prepare_vggflowers.py` with options: 
   
    + `--data_path`: `$DATA_PATH` as above
    + `--label_path`: path of the .mat labels downloaded from the website