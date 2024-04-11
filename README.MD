# Research for Unsupervised Domain Generalisation in Medical Image Segmentation

### GitHub Repository
https://github.com/L1ll3J/4YP


### Requirements
Install all the packages required by running: 
```
pip install -r requirements.txt
```
Exact environments are also included

### Data
We used data from SMDG, A Standardized Fundus Glaucoma Dataset. Download the data from: 

https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset

The format for each labelled database should be:

Labelled/*dataset_name*/images                       - images go here
Labelled/*dataset_name*/masks/both                   - segmentation masks go here


And for unlabelled:

Unlabelled/*dataset_name*/images                     - images go here


NOTE: The provided labelled segmentation masks are seperated into Optic Cup & Optic Disk, for our purposes we want these combined.
This can be done using:
```
python3 4YP/Run/data_aug/combine_cup_and_disk.py
```
### Format:
Each experiment is contained within it's own program to facilitate re-producability. 

All programs are wrapped using wandb's sweep function, parameters can be adjusted by modifying the ```sweep_config``` variable.

There are 3 main experiments; the baseline U-Net, FixMatch & the Adversarial network.


### Baseline U-Net:

To run without data augmentation:
```
cd REFUGE _4YP/Run
python3 BaseLine_NoAugmentation.py
```

To run with online "live" augmentation:
```
cd REFUGE _4YP/Run
python3 BaseLine_LiveAugmentation.py
```

### FixMatch:
The FixMatch paper can be found at:

https://arxiv.org/abs/2001.07685

FixMatch.py is mostly an artifact, but can be used for debugging on a single database.

FixMatchMulti.py is the main program, simply adjust file paths and use:
```
cd REFUGE _4YP/Run
python3 FixMatchMulti.py
```


### Adversarial Network
The conventional implementation of the adversarial network for Unsupervised Domain Adaptation (UDA) (single source / single target) is found in AdvNetwork_Outputs_UDAstyle.py

The adapted implementation of the adversarial network for Unsupervised Domain Generalisation (UDG) (single source / multi target) is found in AdvNetwork_Outputs_FMstyle.py

Both can be run using the same approach as for the baseline U-Net and FixMatch. (no external args required)
# Saving & Results
Currently, since wandb is used, all training metrics / results are logged using the API and can be access using a free Weights and Biases account. Once you're account and environment are configured, use:
```
wandb.login()
```
And provide your API key when prompted.

In addition to this, it is important to modify the ```data_save_path``` variable to ensure you are saving to the correct location. This variable is found twice in every main program, always at the start of the ```train()``` and ```test()``` functions. 
(Make sure to change both!)

After running one of the programs, this location will contain: the saved model - which can be restored at a later date using ```torch.load()``` and some example segmentations so that you can visualise performance.

### To modify this code:
A lot of functions/classes are repeated across programs and so have been stored in custom modules.

The largest and most important custom module is:
```4YP/Run/utils.py```

Every main program uses functions from this module and so care should be taken when making modifications as this will effect all programs.

In addition to this, the base U-Net architecture can be found at:
```4YP/Run/UNET/UNet_model.py```
