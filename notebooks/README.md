# Notebooks

The Jupyter notebooks for the project. Run eda.ipynb first, since it builds the
dataset that model_training.ipynb reads.

## eda.ipynb

Explores and cleans the data, then builds the final modeling table. It loads the
public Zenodo recordings and the extra rooms, checks missing values and duplicates,
drops the badly blank raw columns, melts the 6 snapshots into rows so one profile is
one observation, applies per-room background subtraction, and computes the 7 raw and
7 residual echo features for every profile. It writes
data/processed/final_dataset.csv (131 profiles across 11 rooms, 16 columns).

## model_training.ipynb

Loads data/processed/final_dataset.csv and trains the models. It defines the two
feature sets (raw and residual), makes an 80/20 stratified split, trains Logistic
Regression, Random Forest, and Gradient Boosting on each set, compares them by
accuracy and F1 on the test set, and shows the confusion matrix for the best model.

## eda.html and model_training.html

Rendered copies of the two notebooks, with all cells run, for viewing the outputs
without opening Jupyter.

## Run

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace eda.ipynb
jupyter nbconvert --to notebook --execute --inplace model_training.ipynb
```
