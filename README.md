# Indoor Human Presence Detection with a Smartphone Ultrasonic Sonar

UMBC DATA 606 capstone.

A smartphone plays a short 17 to 22 kHz chirp and records the echo. When a person
is in the room the echo changes. The goal is to tell an empty room from an
occupied one from the echo profile.

## Folders

- data/external_zenodo - the Indoor Acoustic Occupancy dataset (Zenodo 18096214), 3 real rooms
- data/own_data - Data collected by myself in different rooms in the same format, to add more data
- data/processed/final_dataset.csv - the final feature table (131 profiles across 11 rooms)
- notebooks/eda.ipynb - explore and clean the data, then build the final dataset
- notebooks/model_training.ipynb - train and compare the models
- app/streamlit_app.py - a Streamlit app to try the trained model on any reading
- docs - project report, resume, headshot

## Run

Install the packages:

```bash
pip install -r requirements.txt
```

Run the notebooks (run the EDA first, it writes data/processed/final_dataset.csv,
then the model training reads it):

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace eda.ipynb
jupyter nbconvert --to notebook --execute --inplace model_training.ipynb
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```

## Task

Binary classification. The target is occupancy (empty or person). The features are
the echo summary values in data/processed/final_dataset.csv, computed on the raw
profile and on the profile after background subtraction. The best model is a
Random Forest on the after-subtraction features, at about 0.89 accuracy.
