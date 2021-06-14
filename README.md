# Addressing Catastrophic Forgetting in Few-Shot Problems

The code in this repository is for the experiments in [Addressing Catastrophic Forgetting in Few-Shot Problems](https://arxiv.org/abs/2005.00146).

### Main dependencies
+ Python 3.8
+ PyTorch 1.8.0
+ Tensorboard 2.5.0
+ Torchmeta 1.6.0

### Getting started
1. Install PyTorch based on the download information in [PyTorch website](https://pytorch.org/get-started/locally/).
   
1. Install requirements: `pip install -r requirements.txt` and internal modules: `pip install -e .`

1. See `data_prepare` [README](data_prepare/README.md) to prepare for the datasets necessary for this project.

1. A 16GB GPU is sufficient to run any of the experiments, although a smaller GPU might also be possible by setting `cuda_img=false` in the config files.

### Experiments

#### Triathlon and Pentathlon
  
+ Use main file `main_la_seqdataset.py` for LA, `main_vi_seqdataset.py` for VI. 
  
+ Use config files `triathlon_*.json` for triathlon and `pentathlon_*.json` pentathlon. 
  
+ `data_path` takes the path of the parent folder containing all datasets.

    ```
    python train/<MAIN_FILE> --config_path config/<CONFIG_FILE> --data_path <PARENT_DATA_PATH>
    ```
   

#### Omniglot Sequential Task
+ Use main file `main_la_seqtask.py` and config file `omniglot_seqtask_la.json` for LA, `main_vi_seqtask.py` and `omniglot_seqtask_vi.json` for VI. 
  
    ```
    python train/<MAIN_FILE> --config_path config/<CONFIG_FILE> --data_path <PARENT_DATA_PATH>
    ```
