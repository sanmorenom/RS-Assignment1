# Neural Collaborative Filtering (NCF) for MovieLens 1M

This project implements a Neural Collaborative Filtering (NCF) recommender system in PyTorch using the MovieLens 1M dataset.

## Project Overview

The system includes:

- data preprocessing
- NCF model implementation
- training with early stopping
- evaluation with Recall@10 and NDCG@10
- comparison across multiple hyperparameter settings
- validation-loss plotting for the top models

The dataset is treated as implicit feedback:

- ratings >= 4 are considered positive interactions
- negative interactions are generated with negative sampling

## Requirements

- Python 3.10+ recommended
- pip

### Python packages
- torch
- pandas
- numpy
- scikit-learn

### Optional packages
- matplotlib
- scipy

## Dataset

Download the MovieLens 1M dataset and place it in the project folder as:

ml-1m/ratings.dat

Expected structure:

RS-Assignment1/
├── ml-1m/
│   └── ratings.dat
├── processed/
├── checkpoints/
├── model.py
├── Plot.py
├── train.py
├── train_multiple_settings.py
├── Evaluate.py
├── Evaluate_multiple_settings.py
├── preprocessing_UpgradebyG.py
├── README.md
└── .gitignore

Notes:
- `ml-1m/`, `processed/`, and `checkpoints/` are not included in the repository.
- `processed/` and `checkpoints/` will be generated after running the scripts.

## Install dependencies

pip install torch pandas numpy scikit-learn matplotlib scipy

## How to run

### 1. Preprocess the dataset
python preprocessing_UpgradebyG.py

### 2. Train the baseline model
python train.py

### 3. Evaluate the baseline model
python Evaluate.py

### 4. Train multiple model settings
python train_multiple_settings.py

### 5. Evaluate the top saved models
python Evaluate_multiple_settings.py

### 6. Plot validation loss curves for the saved top models
python Plot.py

## File Description

- `preprocessing_UpgradebyG.py`: loads MovieLens 1M, converts ratings to implicit feedback, applies negative sampling, and creates train/validation/test splits.
- `model.py`: defines the NCF architecture with GMF and MLP branches.
- `train.py`: trains the baseline NCF model with early stopping.
- `Evaluate.py`: evaluates the baseline model using Recall@10 and NDCG@10.
- `train_multiple_settings.py`: trains several hyperparameter configurations and keeps the top-performing models.
- `Evaluate_multiple_settings.py`: evaluates the saved top models using full-ranking metrics.
- `Plot.py`: plots validation loss over epochs for the top saved models.

## Notes

- GPU is optional.
- The code automatically uses CUDA if available; otherwise it runs on CPU.
- matplotlib and scipy are only needed for optional plotting.
- Full-ranking evaluation excludes items already seen in training and validation for each user.

## Possible Improvements

Some possible next steps for future work:

- use a user-based split instead of a fully random interaction split
- explore additional hyperparameter settings
- compare different negative sampling ratios
- add more ranking or recommendation quality metrics such as coverage and diversity
- improve experiment tracking and logging
