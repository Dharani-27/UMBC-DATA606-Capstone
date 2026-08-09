# Real-Time Smartphone-Based Human Detection

Detecting whether a room is empty or occupied by using a smartphone as a low-cost,
privacy-preserving sonar.

Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang

### 1. Title and Author

- Project Title: Real-Time Smartphone-Based Human Detection
- Author: Dharani Nadendla, MPS Data Science
- GitHub repository: https://github.com/Dharani-27/UMBC-DATA606-Capstone
- LinkedIn: https://www.linkedin.com/in/dharani-nadendla/
- PowerPoint presentation: to be added
- YouTube video: to be added

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
2. Collected data. 8 additional rooms generated in the exact same file format and
   feature layout, so there is enough data to train a model. Each Collected room
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
