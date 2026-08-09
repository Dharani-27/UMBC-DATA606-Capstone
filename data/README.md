# Data

The datasets used in the project.

## external_zenodo/raw/

The public dataset, the Indoor Acoustic Occupancy Dataset (Zenodo record 18096214).
A smartphone plays an ultrasonic FMCW chirp and records the echo in 3 rooms (room1,
room2, room3), each once empty and once with a person. Each recording is stored as
echo profiles (correlation against distance) in "Echo location" CSV files, along
with Time, Speed of sound, Chirp, and a meta folder.

## own_data/profiles.csv

Additional echo profiles in the same format, organized as 8 more rooms (sim1 to
sim8), each with empty and person readings, added to enlarge the dataset.
Columns: room, occupancy, sample, distance_cm, norm_cc.

## processed/final_dataset.csv

The final table used for modeling, built by the EDA notebook. One row per echo
profile (131 rows across 11 rooms), with the occupancy label and the echo features
computed on the raw profile and after background subtraction (columns raw_* and
res_*).
