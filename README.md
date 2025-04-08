<h1 align="center">
    Attention-aware Ensemble Learning Network (AELNet)
</h1>
<h2 align="center">
    AELNet for Face-Periocular Cross-Modality Matching (FPCM)   
</h2>
<h3 align="center">
    Published in Applied Soft Computing (DOI: 10.1016/j.asoc.2025.113044) </br>
    <a href="https://www.sciencedirect.com/science/article/pii/S1568494625003552"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a>
</h3>
<br/>

![Network Architecture](AEL_Net_Architecture.jpg?raw=true "AELNet")
<br/></br>

## Pre-requisites:
- <b>Environment: </b>Check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command for the environment requirement. These files are slightly filtered manually, so there may be redundant packages.
- <b>Dataset: </b> Download dataset (training and testing) from [this link](https://www.dropbox.com/s/bfub8fmc44tvcxb/periocular_face_dataset.zip?dl=0). Password is _conditional\_biometrics_.
Ensure that datasets are located in `data` directory. Configure `datasets_config.py` file to point to this data directory by changing main path.
- <b>Pre-trained models: </b>(Optional) The pre-trained MobileFaceNet model for fine-tuning or testing can be downloaded from [this link](https://www.dropbox.com/scl/fi/l3k1h3tc12vy7puargfc3/MobileFaceNet_1024.pt?rlkey=m9zock9slmaivhij6sptjyzl6&st=jy9cb6oj&dl=0).

## Training: 
1. Change hyperparameters accordingly in `params.py` file. The set values used are the default, but it is possible to alternatively change them when running the python file.
2. Run `python training/main.py`. The training should start immediately.
3. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

## Testing:
1. Based on the (pre-)trained models in the `models(/pretrained)` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc.
2. Evaluation:
    * Identification / Cumulative Matching Characteristic (CMC) curve: Run `cmc_eval_identification.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate CMC graph.
    * Verification / Receiver Operating Characteristic (ROC) curve: Run `roc_eval_verification.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate ROC graph.
3. Visualization:
    * Gradient-weighted Class Activation Mapping (Grad-CAM): Run `grad_cam.py`, based on the selected images that are stored in a directory. The images will be generated in the `graphs` directory.
    * t-distributed stochastic neighbor embedding (t-SNE) : Run the Jupyter notebook accordingly. Based on the included text file in `data/visualization/tsne/img_lists`, 10 toy identities are selected to plot the t-SNE points, which will be generated in the `graphs` directory.

## Comparison with State-of-the-Art (SOTA) models

| Method | Rank-1 IR (%) <br> (Periocular Gallery) | Rank-1 IR (%) <br> (Face Gallery) | EER (%) <br> (Periocular-Face) |
| --- | --- | --- | --- |
| <a href="https://github.com/tiongsikng/cb_net" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">CB-Net</a> <a href="https://ieeexplore.ieee.org/abstract/document/10201879"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/h5tz21big39wd0dzc70ou/AOabrddckd5cKUF3R2p3jw0?rlkey=l8fksw4ekat5jzcgn66jft6n3&st=t1rayruv&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 77.26 | 68.22 | 9.80 |
| <a href="https://github.com/MIS-DevWorks/HA-ViT" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">HA-ViT</a> <a href="https://ieeexplore.ieee.org/document/10068230"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/crjb30rnxe95e6cdbolsk/AFT0bjj1-OzFuRTrictlAuQ?rlkey=rmpe6mriebl5l051pcfatog11&st=os5z2084&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 64.72 | 64.01 | 13.14 |
| <a href="https://github.com/tiongsikng/gc2sa_net" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">GC<sup>2</sup>SA-Net</a> <a href="https://ieeexplore.ieee.org/document/10418204"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/j7tfsk61jz6dch8hyl1hp/h?rlkey=b22nw4ff5kelu5ivti7ioy1mr&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 90.77 | 88.93 | 6.50 |
| <a href="https://github.com/MIS-DevWorks/FBR" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">MFA-ViT</a> <a href="https://ieeexplore.ieee.org/document/10656777"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/1guujtv39cpktxk6dknve/ADx9ow2FbTTRMLFGtoKU-yM?rlkey=ooxn4uzruiwrmmdo5izbjuzyn&st=25c1acfu&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 89.34 | 85.35 | 9.41 |
| <a href="https://github.com/tiongsikng/ael_net" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">AELNet</a> <a href="https://www.sciencedirect.com/science/article/pii/S1568494625003552"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/pwc3gnu6vggrtbfwk9vw1/AITjo9pNnqVs8HXfOY2tSGY?rlkey=qujqfhtadnvcxp00zr75nj10m&st=famfx1am&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 92.47 | 90.71 | 6.31 |

### The project directories are as follows:
<pre>
├── configs: Dataset path configuration file and hyperparameters.
│   ├──  datasets_config.py - Contains directory path for dataset files. Change 'main' in 'main_path' dictionary to point to dataset, e.g., <code>/home/ael_net/data</code> (without slash).
│   └──  params.py - Adjust hyperparameters and arguments in this file for training. 
├── data: Dataloader functions and preprocessing.
│   ├──  <i><b>[INSERT DATASET HERE.]</i></b>
│   └──  data_loader.py - Generate training and testing PyTorch dataloader. Adjust the augmentations etc. in this file. Batch size of data is also determined here, based on the values set in <code>params.py</code>.
├── eval: Evaluation metrics (identification and verification). Also contains CMC and ROC evaluations.
│   ├──  cmc_eval_identification.py - Evaluates Rank-1 Identification Rate (IR) and generates Cumulative Matching Characteristic (CMC) curve, which are saved as <code>.pt</code> files in <code>data</code> directory. Use these <code>.pt</code> files to generate CMC curves.
│   ├──  plot_cmc_roc_sota.ipynb - Notebook to plot CMC and ROC curves side-by-side, based on generated <code>.pt</code> files from <code>cmc_eval_identification.py</code> and <code>roc_eval_verification.py</code>. Graph is generated in <code>graphs</code> directory.
│   ├──  plot_tSNE.ipynb - Notebook to plot t-SNE images based on the 10 identities of periocular-face toy examples. Example of text file (which correlates to the image paths) are in <code>data/visualization/tsne/img_lists</code>.
│   └──  roc_eval_verification.py - Evaluates Verification Equal Error Rate (EER) and generates Receiver Operating Characteristic (ROC) curve, which are saved as <code>.pt</code> files in <code>data</code> directory. Use these <code>.pt</code> files to generate ROC curves.
├── graphs: Directory where graphs are generated.
│   └──  <i>CMC and ROC curve file is generated in this directory. Some evaluation images are also generated in this directory.</i>
├── logs: Directory where logs are generated.
│   └──  <i>Logs will be generated in this directory. Each log folder will contain backups of training files with network files and hyperparameters used.</i>
├── models: Directory to store pre-trained models. Trained models are also generated in this directory.
│   ├──  <i><b>[INSERT PRE-TRAINED MODELS HERE.]</i></b>
│   ├──  <i>Trained models will also be stored in this directory.</i>
│   └──  <i><b>The base MobileFaceNet for fine-tuning the AELNet can be downloaded in <a href="https://www.dropbox.com/scl/fi/l3k1h3tc12vy7puargfc3/MobileFaceNet_1024.pt?rlkey=m9zock9slmaivhij6sptjyzl6&e=1&st=jy9cb6oj&dl=0">this link</a>.</i></b>
├── network: Contains loss functions and network related files.
│   ├──  ael_net.py - Architecture file for AELNet.
│   ├──  load_model.py - Loads pre-trained weights based on a given model.
│   └──  logits.py - Contains some loss functions that are used.
└── <i>training:</i> Main files for training AELNet.
    ├──  main.py - Main file to run for training. Settings and hyperparameters are based on the files in <code>configs</code> directory.
    └──  train.py - Training file that is called from <code>main.py</code>. Gets batch of dataloader and contains criterion for loss back-propagation.
</pre>

#### Citation for this work:
```
@ARTICLE{ael_net,
title = {Attention-aware ensemble learning for face-periocular cross-modality matching},
journal = {Applied Soft Computing},
volume = {175},
pages = {113044},
year = {2025},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2025.113044},
author = {Tiong-Sik Ng and Andrew Beng Jin Teoh}
}
```