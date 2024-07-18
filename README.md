# Attention-aware Ensemble Learning Network (AELNet)
## AELNet for Face-Periocular Cross-Modality Matching (FPCM)

![Network Architecture](AEL_Net_Architecture.jpg?raw=true "AELNet")

The project directories are as follows:

- configs: Dataset path configuration file and hyperparameters.
    * datasets_config.py - Directory path for dataset files. Change 'main' in 'main_path' dictionary to point to dataset, e.g., `/home/tiongsik/ael_net/data` (without slash).
    * params.py - Adjust hyperparameters and arguments in this file for training. 
- data: Dataloader functions and preprocessing.
    * __**INSERT DATASET HERE**__
    * data_loader.py - Generate training and testing PyTorch dataloader. Adjust the augmentations etc. in this file. Batch size of data is also determined here, based on the values set in `params.py`.
- eval: Evaluation metrics (identification and verification). Also contains CMC and ROC evaluations.
    * cmc_eval.py - Evaluates and generates Cumulative Matching Characteristic (CMC) curve, which are saved as `.pt` files in `data` directory. Use these `.pt` files to generate CMC curves.
    * identification.py - Evaluates Rank-1 Identification Rate (IR).
    * plot_cmc_roc_sota.ipynb - Notebook to plot CMC and ROC curves side-by-side, based on generated `.pt` files from `cmc_eval.py` and `roc_eval.py`. Graph is generated in `graphs` directory.
    * roc_eval.py - Evaluates and generates Receiver Operating Characteristic (ROC) curve, which are saved as `.pt` files in `data` directory. Use these `.pt` files to generate ROC curves.
    * verification.py - Evaluates Verification Equal Error Rate (EER).
- graphs: Directory where graphs are generated.
    * _CMC and ROC curve file is generated in this directory._
- logs: Directory where logs are generated.
    * _Logs will be generated in this directory. Each log folder will contain backups of training files with network files used._
- models: Directory to store pre-trained models. Trained models are also generated in this directory.
    * __**INSERT PRE-TRAINED MODELS HERE. The base MobileFaceNet for fine-tuning the AELNet can be downloaded in [this link](https://www.dropbox.com/scl/fo/sx61beaupkwa1574fst2z/h?rlkey=onwf8vji3h20og0w7s6sxznlc&dl=0).**__
    * _Trained models will also be stored in this directory._
- network: Contains loss functions and network related files.
    * `facexzoo_network` - Directory contains architecture files from FaceXZoo.
    * `SOTA` - Directory contains architecture files that are used for State-of-the-Art (SOTA) comparison, namely MFA-ViT, HA-ViT, and CMB-Net. _Since MFA-ViT and HA-ViT has their own data loader and has a slightly different setting, the `MFA-ViT` or `HA_ViT` directoriess contain their own data loader and evaluation files for simplicity._
    * ael_net.py - Architecture file for AELNet.
    * load_model.py - Loads pre-trained weights based on a given model.
    * logits.py - Contains some loss functions that are used.
- __training:__ Main files for training AELNet.
    * main.py - Main file to run for training. Settings and hyperparameters are based on the files in `configs` directory.
    * train.py - Training file that is called from `main.py`. Gets batch of dataloader and contains criterion for loss back-propagation.

### Pre-requisites (requirements):
Check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command. These files are partially filtered manually, so there may be redundant packages.
Download dataset (training and testing) from [this link](https://www.dropbox.com/s/bfub8fmc44tvcxb/periocular_face_dataset.zip?dl=0). Password is _conditional\_biometrics_.

### Training:
1. Ensure that datasets are located in `data` directory. Configure `datasets_config.py` file to point to this data directory.
2. Change hyperparameters accordingly in `params.py` file. The set values used are the default.
3. Run `main.py` file. The training should start immediately.
4. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

### Testing:
0. Pre-trained models for fine-tuning can be downloaded from [this link](https://www.dropbox.com/s/g8gn4x4wp0svyx5/pretrained_models.zip?dl=0). Password is _conditional\_biometrics_.
1. Based on the (pre-)trained models in the `models(/pretrained)` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc.
2. Evaluation:
    * Cumulative Matching Characteristic (CMC) curve: Run `cmc_eval.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate CMC graph.
    * Identification: Run `identification.py`.
    * Receiver Operating Characteristic (ROC) curve: Run `roc_eval.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate ROC graph.
    * Verification: Run `verification.py`.

### Comparison with State-of-the-Art (SOTA) models

| Method | Rank-1 IR (%) <br> (Periocular Gallery) | Rank-1 IR (%) <br> (Face Gallery) | EER (%) <br> (Periocular-Face) |
| --- | --- | --- | --- |
| [CB-Net](https://www.dropbox.com/scl/fi/s00q4vhs0sqb3wybl7gft/CB-Net.pth?rlkey=aczpmgt27cyms78s0vxlc8zqm&st=ahjr50d5&dl=0) | 77.26 | 68.22 | 9.80 |
| [HA-ViT](https://www.dropbox.com/scl/fi/954fltr100zvwjyyl9fn5/HA-ViT.pth?rlkey=pip3eydsib2o11uefhluhoy2e&st=8irhf1ew&dl=0) | 64.72 | 64.01 | 13.14 |
| [GC<sup>2</sup>SA-Net](https://www.dropbox.com/scl/fi/wl0yut0fr7rvgpp4d3izx/GC2SA-Net.pth?rlkey=ui9feg6ty5ip4ebkds9ggiflm&st=g2l20tso&dl=0) | 90.77 | 88.93 | 6.50 |
| [MFA-ViT](https://www.dropbox.com/scl/fi/ate7icoe714pw3592mtmm/MFA-ViT.pth?rlkey=oa2iapqlwomwivqzppzftl8ff&st=wc8iwepz&dl=0) | 89.34 | 85.35 | 9.41 |
| [AELNet](https://www.dropbox.com/scl/fo/j90nx00akg0bkp0dr7a6w/h?dl=0&rlkey=1k8eae7r7lbt326kzobgy88fl) | 92.47 | 90.71 | 6.31 |
