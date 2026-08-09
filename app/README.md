# App

A small Streamlit app to interact with the trained model.

Pick a room and an echo reading, and the app shows the echo profile and whether the
model thinks the room is empty or has a person, with a confidence bar and whether it
matched the true label. It uses the Random Forest on the residual features (after
background subtraction), the same model as the model training notebook. Predictions
are out-of-fold, so the model did not train on the reading it is judging.

## Run
From the project root:

    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

It opens in your browser.
