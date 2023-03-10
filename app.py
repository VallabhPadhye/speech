import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load Keras model
@st.cache(allow_output_mutation=True)
def load_emotion_model():
    model = load_model('emotion_model.h5')
    return model

emotion_model = load_emotion_model()

# Define function to extract features from audio file
def extract_feature(file_path):
    try:
        audio_data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None 
     
    return mfccs_processed

# Define emotions
emotions = {
    '0': 'Angry',
    '1': 'Calm',
    '2': 'Fearful',
    '3': 'Happy',
    '4': 'Neutral',
    '5': 'Sad'
}

# Define app layout
st.set_page_config(page_title="Voice Emotion Recognition App")
st.title("Voice Emotion Recognition App")

# Add file uploader to sidebar
uploaded_file = st.sidebar.file_uploader("Upload an audio file", type=['wav'])

# If file is uploaded, extract features and predict emotion
if uploaded_file is not None:
    st.sidebar.audio(uploaded_file, format='audio/wav')
    feature = extract_feature(uploaded_file)
    feature = np.expand_dims(feature, axis=0)
    prediction = emotion_model.predict(feature).argmax(axis=1)[0]
    st.write("The predicted emotion is:", emotions[str(prediction)])
