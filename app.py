import streamlit as st

import librosa

import numpy as np

import tensorflow as tf

import sounddevice as sd

# Load the trained model

model = tf.keras.models.load_model('path/to/your/trained/model')

# Define the labels for different emotions

labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define a function to extract features from the audio data

def extract_feature(file_name, mfcc, chroma, mel):

    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    features = np.array([])

    if chroma:

        chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)

        features = np.hstack((features, chroma))

    if mfcc:

        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

        features = np.hstack((features, mfccs))

    if mel:

        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

        features = np.hstack((features, mel))

    return features

# Define the Streamlit app

def app():

    st.title("Speech Emotion Recognition")

    # Record audio from the user's microphone

    st.write("Press the button to start recording")

    duration = st.slider("Recording duration (in seconds)", 1, 10, 3, 1)

    with st.spinner('Recording...'):

        recording = sd.rec(int(duration * 22050), samplerate=22050, channels=1)

        sd.wait()

        st.write("Recording complete")

    # Extract features from the recorded audio

    features = extract_feature(recording, mfcc=True, chroma=True, mel=True)

    # Normalize the features

    features = (features - np.mean(features)) / np.std(features)

    # Make the prediction

    prediction = model.predict(np.expand_dims(features, axis=0))

    prediction_label = labels[np.argmax(prediction)]

    # Display the result

    st.write(f"Predicted emotion: {prediction_label}")

# Run the app

if __name__ == '__main__':

    app()

