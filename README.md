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


### Experiments

#### Triathlon and Pentathlon
  
+ Use main file `main_la_seqdataset.py` for BOMLA, `main_vi_seqdataset.py` for BOMVI. 
  
+ Use config files `triathlon_*.json` for triathlon and `pentathlon_*.json` pentathlon. 
  
+ `data_dir` takes the path of the parent folder containing all datasets.

    ```
    python train/<MAIN_FILE> --config_path config/<CONFIG_FILE> --data_dir <PARENT_DATA_DIR>
    ```
   

#### Omniglot Sequential Task
+ Use main file `main_la_seqtask.py` and config file `omniglot_seqtask_la.json` for BOMLA, `main_vi_seqtask.py` and `omniglot_seqtask_vi.json` for BOMVI. 
    ```
    python train/<MAIN_FILE> --config_path config/<CONFIG_FILE> --data_dir <PARENT_DATA_DIR>
    ```
