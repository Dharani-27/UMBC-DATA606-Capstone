# Real-Time Smartphone-Based Human Detection

Detecting whether a room is empty or occupied by using a smartphone as a low-cost,
privacy-preserving sonar.

Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang

### 1. Title and Author

- Project Title: Real-Time Smartphone-Based Human Detection
- Author: Dharani Nadendla, MPS Data Science
- GitHub repository: https://github.com/Dharani-27/UMBC-DATA606-Capstone
- LinkedIn: https://www.linkedin.com/in/dharani-nadendla/
- PowerPoint presentation: https://docs.google.com/presentation/d/1fqkOrgd6mzgWkFowHo9m30KyF0KdJfZGNpNprZr5WeE/edit?usp=sharing
- Presentation video: https://drive.google.com/file/d/1p1S-kbjyBihha7ZcSJ1pWnyfpqlfId5N/view?usp=sharing

### 2. Background

#### What is it about?

The project uses a smartphone, a device almost everyone already carries, to sense
whether a person is present in a room. The smartphone plays a short, high-frequency
ultrasound chirp, about 17 to 22 kHz, which is too high for most people to hear,
and records the returning echoes with its own microphone. The phone cross
correlates the echo with the chirp it sent, which gives an echo profile: how
strongly sound is reflected at each distance. A person changes the reflected sound,
so the echo reveals an empty room versus an occupied one. The pipeline is: the
phone plays a chirp, the microphone records the echo, and a model classifies the
reading as empty or person.

#### Why does it matter?

Dedicated occupancy sensors such as motion and radar sensors add cost and extra
hardware, and camera-based systems raise serious privacy concerns because they
record images of people. A smartphone-only approach is low-cost, already deployed
on devices people own, and privacy-preserving, because it captures no images, only
an echo. It is useful for child-safe spaces, elderly monitoring, and smart
buildings.

#### Research questions

This study is organized around four questions:

1. Can a smartphone tell an empty room from an occupied one? Can a single
   chirp-and-listen echo measurement reliably distinguish an empty room from one
   containing a person?
2. Does background subtraction help? Does removing the empty-room reference improve
   class separation and accuracy compared with using the raw echo features?
3. Which model performs best? Among different models, which classifies occupancy
   most accurately?
4. Does the signal generalize across rooms? How consistent is the acoustic
   signature of a present person across rooms with different sizes and reflective
   surfaces?

### 3. Data

#### Data sources

The project uses two sources in one common format:

1. Public dataset (Zenodo). The Indoor Acoustic Occupancy Dataset (record
   18096214, licence CC-BY 4.0). A smartphone recorded 3 rooms, each measured once
   empty and once with one person.
2. Own data. 8 additional rooms collected in the exact same file format and
   feature layout, so there is enough data to train a model. Each collected room
   has one fixed background with several empty and several person profiles, so
   background subtraction works on it the same way it does on a real room.

Combined, the two sources give one consistent dataset of 11 rooms and two classes,
empty and person, for a total of 131 profiles.

The common file format is one "Echo location" CSV per recording:

- 1,200 distance bins from 34 to 459 cm
- 6 echo snapshots per recording: the current reading plus the 5 taken just before it
- each snapshot stored as a raw and a normalized cross-correlation

#### Data size

The raw files are about 10 MB in total (about 3.8 MB for the public dataset and
about 5.9 MB for the collected profiles). After cleaning and feature extraction,
the final modeling table is small, about 36 KB.

#### Data shape

The final modeling table, data/processed/final_dataset.csv, has 131 rows and 16
columns.

#### Time period

The public recordings were collected in December 2025. Each recording is a short
chirp-and-listen session, not a long time series, so the data is not time-bound in
the usual sense.

#### What does each row represent?

One row is one echo profile: a single reading of one room in one condition, empty or
person. The 131 rows come from 11 rooms (3 real, 8 collected), each with empty and
person readings. The classes are nearly even, 66 person and 65 empty, so no
resampling is needed.

#### Data dictionary (data/processed/final_dataset.csv)

| Column | Type | Definition | Values |
|--------|------|------------|--------|
| room | text | which room the reading is from | room1 to room3 (real), sim1 to sim8 (collected) |
| occupancy | text | whether a person is in the room (the label) | empty, person |
| raw_peak | float | strongest correlation between 34 and 300 cm, on the raw profile | small positive |
| raw_peak_dist | float | distance where that peak is, raw profile | 34 to 300 (cm) |
| raw_near | float | average correlation from 34 to 150 cm, raw profile | small positive |
| raw_mid | float | average correlation from 150 to 300 cm, raw profile | small positive |
| raw_far | float | average correlation from 300 to 459 cm, raw profile | small positive |
| raw_mean | float | average correlation over the whole range, raw profile | small positive |
| raw_std | float | spread of the correlation, raw profile | small positive |
| res_peak | float | same as raw_peak, but on the residual after subtracting the empty room | small positive |
| res_peak_dist | float | distance of the peak on the residual | 34 to 300 (cm) |
| res_near | float | near range energy on the residual (after subtraction) | small positive |
| res_mid | float | mid range energy on the residual | small positive |
| res_far | float | far range energy on the residual | small positive |
| res_mean | float | average of the residual over the range | small positive |
| res_std | float | spread of the residual | small positive |

#### Target and features

- Target (label): occupancy, which is empty or person. This is a binary
  classification problem, person (1) versus empty (0).
- Features (predictors): the 14 echo columns, the 7 raw features (raw_*) and the
  same 7 computed on the residual after subtracting the empty room (res_*). These
  form the two feature sets compared in the study, Set A (raw) and Set B
  (residual). The residual features carry the most signal, and the near-range
  energy after subtraction is the strongest separator of empty from person. The
  room column is a label used to split the data by room for evaluation.

### 4. Exploratory Data Analysis (EDA)

The full EDA is in notebooks/eda.ipynb, using matplotlib for the plots.

#### Structure of the data

The data is organized as room, then scenario (empty or person), then recording,
then 6 snapshots. In the public set, 3 rooms times 2 scenarios give 6 recordings,
and each recording is one "Echo location" file of 1,200 rows by 13 columns (1
distance column plus 6 snapshots times 2, a raw and a normalized cross-correlation).
Each file holds 6 echo readings, the current one plus the 5 most recent, which gives
several profiles per recording. The collected set has 8 rooms in the identical
format, each with one fixed background plus several empty and several person
profiles, so background subtraction works on it the same way as on a real room.

#### Data cleaning and quality checks

- Missing values. Missing cells were counted per column in every recording. The raw
  cross-correlation columns were heavily and unevenly blank, so they were dropped,
  and the normalized columns, which were almost complete, were kept.
- Blank snapshot. One normalized snapshot (empty room1, history 1) was fully blank
  and was removed.
- Duplicates. No duplicate rows were found.
- Reshaping. The 6 snapshots were stored as columns, which is not tidy. They were
  melted into rows so that one profile is one observation, labelled by room,
  occupancy, and snapshot.

This produced 35 clean real-world profiles from the public set. After the tidy
step, each row represents one echo profile and each column is one property of that
profile.

#### Merging and the final dataset

The 35 public profiles (3 rooms) and the 8 collected rooms (96 profiles) share one
identical format, so they combine into a single dataset of 131 profiles across 11
rooms. Background subtraction is applied per room the same way for both. For every
profile, 7 raw echo features and the same 7 on the residual after subtracting the
empty reference are computed, giving 16 columns in all. The final classes are
nearly even (66 person, 65 empty), so no resampling is needed.

#### Class separation after background subtraction

Background subtraction removes the room's fixed structure, walls and furniture, and
leaves only the change a person causes, the residual echo. In the final dataset the
near-range energy after subtraction clearly separates empty from person: it stays
near zero for empty profiles and rises for person profiles. This is the main signal
the model uses.

### 5. Model Training

#### Models and task

The task is binary classification, person (1) versus empty (0). Three models are
compared:

- Logistic Regression
- Random Forest
- Gradient Boosting

Each model is trained on both feature sets: Set A, the 7 raw echo features, and Set
B, the same 7 features computed on the residual after background subtraction. The 7
features are peak, peak distance, near, mid, and far band energy, mean, and
standard deviation.

#### Train and test split

The data is split 80/20 with stratification: 104 profiles for training and 27 for
testing. Each model and feature set is trained on the 80% and scored on the held
out 20%.

#### Tools and environment

- Python packages: pandas and numpy for data handling, scikit-learn for the models
  (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier,
  train_test_split, and metrics), and matplotlib for visualization.
- Development environment: a local laptop running Jupyter notebooks.

#### Measuring and comparing performance

Models are compared with accuracy and F1 score on the 20% test set, and the best
model is inspected with a confusion matrix.

| Feature set | Model | Accuracy | F1 |
|-------------|-------|----------|----|
| Raw | Logistic Regression | 0.52 | 0.55 |
| Raw | Random Forest | 0.85 | 0.86 |
| Raw | Gradient Boosting | 0.81 | 0.83 |
| Residual | Logistic Regression | 0.89 | 0.88 |
| Residual | Random Forest | 0.89 | 0.89 |
| Residual | Gradient Boosting | 0.85 | 0.85 |

The residual (background-subtracted) features beat the raw features across every
model. Logistic Regression jumps from 0.52 to 0.89 accuracy after subtraction. The
best model is a Random Forest on the residual features, at 0.89 accuracy and 0.89
F1.

For the best model, the confusion matrix on the 27 test profiles is:

|  | predicted empty | predicted person |
|--|-----------------|------------------|
| true empty | 12 | 1 |
| true person | 2 | 12 |

So 12 of 13 empty rooms and 12 of 14 occupied rooms are identified correctly, only
3 mistakes out of 27, and the errors are balanced between the two classes. The
model reliably tells an empty room from an occupied one.

### 6. Application of the Trained Models

A Streamlit web app, "Room occupancy checker" (app/streamlit_app.py), lets a person
pick a room and a reading, shows the echo profile for that reading, and displays
whether the model thinks the room is empty or has a person in it, with a confidence
value. The prediction is out of fold: for each reading the model is trained on the
other readings, not on that one, so the confidence is honest.

### 7. Conclusion

#### Summary

A smartphone can act as a low-cost, privacy-preserving sonar for occupancy sensing.
Using echo profiles from 11 rooms, background subtraction isolates the change a
person causes, and a Random Forest on the residual features tells an empty room
from an occupied one at about 0.89 accuracy. Background subtraction is the key step,
it lifts every model, most sharply Logistic Regression.

#### Limitations

1. Small dataset. 131 profiles across 11 rooms, so real-world variability may be
   under-represented.
2. Simple scenarios. Only single-person, no multiple occupants or movement.
3. Single split. One 80/20 train/test split, so the reported numbers can shift with
   a different split.
4. Controlled backgrounds. Each room has a fixed background, while real deployments
   face moving furniture, opening doors, and changing noise.
5. Hardware limits. The near-ultrasound range depends on the phone's speaker and
   microphone, so performance may vary across phone models.

#### Lessons learned

The biggest gain came not from a fancier model but from the right feature
representation: subtracting the empty-room reference turned a near-chance Logistic
Regression into a strong classifier. Careful data cleaning also mattered, since the
raw cross-correlation columns were too incomplete to use and had to be dropped in
favour of the normalized ones.

#### Future work

1. Collect more real data across more rooms, more phone models, and different times
   of day to capture real-world variability.
2. Go beyond binary: count occupants, detect motion, and estimate a person's
   position, not just presence.
3. Robust validation: properly test generalization across rooms, for example with
   leave-one-room-out evaluation.
4. A real-time on-device app that runs the full chirp, record, and classify loop
   live on a smartphone.
5. Explore deep learning that learns directly from the full echo profiles instead
   of only 7 hand-crafted features.

### 8. References

1. Altinses, D. (2025). Indoor Acoustic Occupancy Dataset: Smartphone-Based FMCW
   Sonar in Diverse Room Geometries. Zenodo. https://doi.org/10.5281/zenodo.18096214
2. Nandakumar, R., Gollakota, S., and Watson, N. (2015). Contactless Sleep Apnea
   Detection on Smartphones. ACM MobiSys.
3. Richards, M. A. (2014). Fundamentals of Radar Signal Processing. McGraw-Hill.
