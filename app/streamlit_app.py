"""
Room occupancy checker.

Pick an echo reading and the app shows the profile and whether the model thinks
the room is empty or has a person. It uses the Random Forest on the residual
features (after background subtraction), the same model as the model training
notebook. Run with:  streamlit run app/streamlit_app.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "data", "external_zenodo", "raw")
SIM_FILE = os.path.join(HERE, "..", "data", "own_data", "profiles.csv")
REAL_ROOMS = ["room1", "room2", "room3"]

DIST = np.linspace(34.0, 458.65, 1200)
BODY = (DIST >= 34) & (DIST <= 300)


def seven(curve):
    """The 7 residual features, same as the notebooks."""
    a = np.abs(curve)
    i = int(a[BODY].argmax())
    return {
        "res_peak": a[BODY][i], "res_peak_dist": DIST[BODY][i],
        "res_near": a[(DIST >= 34) & (DIST <= 150)].mean(),
        "res_mid": a[(DIST > 150) & (DIST <= 300)].mean(),
        "res_far": a[(DIST > 300) & (DIST <= 459)].mean(),
        "res_mean": a.mean(), "res_std": a.std(),
    }


def load_real(occ, room):
    f = pd.read_csv(os.path.join(BASE, occ, room, "Echo location.csv"), sep=";")
    cols = [c for c in f.columns if c.startswith("Normalized CC")]
    snaps = [pd.to_numeric(f[c], errors="coerce").to_numpy() for c in cols]
    return [s for s in snaps if not np.isnan(s).all()]


@st.cache_data
def load_everything():
    """Load every reading, compute residual features, and get an honest
    out-of-fold prediction for each reading (from a model that did not train on it)."""
    sim = pd.read_csv(SIM_FILE)
    sim_rooms = sorted(sim.room.unique(), key=lambda x: int(x[3:]))

    catalog, feat_rows = [], []
    for room in REAL_ROOMS + sim_rooms:
        if room in REAL_ROOMS:
            E, P = load_real("empty", room), load_real("person", room)
        else:
            s = sim[sim.room == room]
            E = [s[(s.occupancy == "empty") & (s["sample"] == x)].sort_values("distance_cm").norm_cc.values
                 for x in s[s.occupancy == "empty"]["sample"].unique()]
            P = [s[(s.occupancy == "person") & (s["sample"] == x)].sort_values("distance_cm").norm_cc.values
                 for x in s[s.occupancy == "person"]["sample"].unique()]

        ref = np.mean(E, axis=0)
        noise = np.mean([np.mean(np.abs(E[i] - np.mean([E[j] for j in range(len(E)) if j != i], axis=0))[BODY])
                         for i in range(len(E))])
        rid = 0
        for i_e, e in enumerate(E):
            ref_i = np.mean([E[j] for j in range(len(E)) if j != i_e], axis=0)
            catalog.append({"room": room, "reading": rid, "occupancy": "empty", "curve": e})
            feat_rows.append(seven((e - ref_i) / noise)); rid += 1
        for p in P:
            catalog.append({"room": room, "reading": rid, "occupancy": "person", "curve": p})
            feat_rows.append(seven((p - ref) / noise)); rid += 1

    X = pd.DataFrame(feat_rows)
    y = np.array([1 if c["occupancy"] == "person" else 0 for c in catalog])
    proba = cross_val_predict(RandomForestClassifier(n_estimators=300, random_state=0),
                              X, y, cv=5, method="predict_proba")
    return catalog, proba[:, 1]     # probability of person for each reading


# ---------------- the page ----------------
st.title("Room occupancy checker")
st.write("A smartphone plays an ultrasonic chirp and records the echo. This app takes "
         "one echo reading and tells you whether the model thinks the room is empty "
         "or has a person in it.")

catalog, person_prob = load_everything()
rooms = sorted({c["room"] for c in catalog},
               key=lambda r: (r in REAL_ROOMS and 0 or 1, r))

def room_label(r):
    return r.replace("sim", "own") if r.startswith("sim") else r

room = st.selectbox("Room", rooms, format_func=room_label)
items = [(i, c) for i, c in enumerate(catalog) if c["room"] == room]
choice = st.selectbox("Reading", range(len(items)),
                      format_func=lambda k: f"reading {items[k][1]['reading'] + 1}")
gi, entry = items[choice]

# echo profile
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(DIST, entry["curve"], color="black")
ax.set_xlabel("distance (cm)"); ax.set_ylabel("correlation")
ax.set_title("Echo profile"); ax.grid(alpha=0.3)
st.pyplot(fig)

# prediction
p = float(person_prob[gi])
pred = "person" if p >= 0.5 else "empty"
conf = p if pred == "person" else 1 - p

if pred == "person":
    st.subheader("Prediction: person detected")
else:
    st.subheader("Prediction: room is empty")
st.write(f"confidence: {conf * 100:.0f}%")
st.progress(conf)

truth = entry["occupancy"]
if pred == truth:
    st.success(f"Correct. The true label is {truth}.")
else:
    st.error(f"Wrong. The true label is {truth}.")
