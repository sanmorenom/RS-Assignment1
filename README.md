# Neural Collaborative Filtering (NCF) for MovieLens 1M

This project implements a Neural Collaborative Filtering (NCF) recommender system in PyTorch using the MovieLens 1M dataset.

## Project Overview

The system includes:

- data preprocessing
- NCF model implementation
- training with early stopping
- evaluation with Recall@10 and NDCG@10

The dataset is treated as implicit feedback:

- ratings **>= 4** are considered positive interactions
- negative interactions are generated with negative sampling

---

## Project Structure

```text
RS-Assignment1/
├── ml-1m/                      # MovieLens 1M dataset (not included in repo)
├── processed/                  # Generated processed files (not included in repo)
├── checkpoints/                # Saved trained model (not included in repo)
├── model.py                    # NCF model
├── train.py                    # Training script
├── Evaluate.py                 # Evaluation script
├── preprocessing_UpgradebyG.py # Preprocessing script
├── README.md
└── .gitignore


## Possible Improvements

Some possible next steps for future work:

- **User-based split:** use a split per user instead of a fully random interaction split.
- **Hyperparameter tuning:** try different values for:
  - GMF dimension: `32`, `64`, `128`
  - MLP layers: [128, 64, 32, 16],[64, 32, 16, 8], [128,64, 32, 16, 8]
  - dropout: `0.05`, `0.1`, `0.2`
  - learning rate: `0.0001`, `0.001`, `0.00001`
- **Negative sampling:** compare different negative sampling ratios such as `1`, `3`, `5`.
- **Experiment tracking:** save best epoch, validation loss, Recall@10, and NDCG@10 for each run.
- **Training logs:** store train/validation loss per epoch for easier plotting and analysis.
- **Model comparison:** train several settings and compare results in one table.
- **Cleaner naming:** rename `Evaluate.py` to `evaluate.py` and `preprocessing_UpgradebyG.py` to `preprocessing.py`.


(Basically use your gpu and do it, I did all this with only CPU because it was too late therefore the SSH was not available probably they turn them off in the nights...)
