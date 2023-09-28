# EEGTrust
This repository contains the code implementation used in the paper Real-time Trust Recognition in Human-Robot Cooperation Using EEG, which has been submited at ICRA2024.

<!-- If you find this repo helpful for your research, please cite our paper using the following reference: -->

# Contents
- [EEGTrust](#eegtrust)
- [Contents](#contents)
- [1. Updates](#1-updates)
- [2. Download Dataset](#2-download-dataset)
- [3. Scripts](#3-scripts)
  - [3.1 Environments](#31-environments)
  - [3.2 Data Pre-processing](#32-data-pre-processing)
  - [3.3 Training and Evaluation](#33-training-and-evaluation)
- [4. Thanks](#4-thanks)


# 1. Updates
- 26/9/2023 EEGTrust v1.0 Uploaded.
  
# 2. Download Dataset
Our dataset can be downloaded via https://yunpan.tongji.edu.cn/link/AAD7EAE6EAF0A4424DA60F4E5522230CEB. Detailed instructions for utilizing the dataset can be found in the REDEME file provided on the data access portal page. If there are any questions about the dataset, please contact email xucaiyue@tongji.edu.cn.

# 3. Scripts
## 3.1 Environments
To create a python environments to use the scripts in this repository, run the following command:
```
conda env create -f scripts/EEGTrust.yaml -n EEGTrust
```

## 3.2 Data Pre-processing
Our data preprocessing was done in Matlab EEGLAB and consisted mainly of re-reference (common average reference), filtering (band-pass and notch filtering), extract epochs and remove baseline, and independent component analysis (ICA) to remove eye movement artifacts.

## 3.3 Training and Evaluation
In this repository we provide our EEG Transformer and several baseline models for training our EEG-based trust recognition models, including NB, KNN, SVM, CNN. For each model, we provide both slice-wise and trial-wise cross-validation scripts.
To perform slice-wise cross-validation of Transformer, run the following command:
```
python trust_transformer.py
```
To perform trial-wise cross-validation of Transformer, runt the following command:
```
python trust_transformer_cross_trial.py
```
# 4. Thanks
Special thanks to the authors of the [EEGTorch](https://torcheeg.readthedocs.io/en/latest/), whose excellent code was used as a basis for the generation scripts in the repository.