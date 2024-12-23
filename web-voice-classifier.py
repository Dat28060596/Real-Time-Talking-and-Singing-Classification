import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pyaudio
import time
import base64
from PIL import Image
from io import BytesIO


# Load the trained model (ensure the path is correct)
model = tf.keras.models.load_model(r"C:\Users\Admin\PycharmProjects\MLAlgorithms\intro to cs\bestmodel.keras")

# Initialize audio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, input=True, frames_per_buffer=22050)

# Add custom CSS for a background image with zoom functionality
def add_background_image_from_upload(uploaded_file, zoom_level):
    """
    Allow the user to upload and crop an image to use as the background.
    Includes zoom functionality.
    """
    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Allow user to crop the image
        st.sidebar.markdown("### Crop the Image")
        left = st.sidebar.slider("Left", 0.0, 1.0, 0.0)
        top = st.sidebar.slider("Top", 0.0, 1.0, 0.0)
        right = st.sidebar.slider("Right", 0.0, 1.0, 1.0)
        bottom = st.sidebar.slider("Bottom", 0.0, 1.0, 0.0)

        # Calculate crop box and crop the image
        width, height = image.size
        crop_box = (
            int(left * width),
            int(top * height),
            int(right * width),
            int(bottom * height)
        )
        cropped_image = image.crop(crop_box)

        # Convert cropped image to base64
        buffer = BytesIO()
        cropped_image.save(buffer, format="PNG")  # Save as PNG
        encoded_image = base64.b64encode(buffer.getvalue()).decode()

        # Inject CSS for the background with zoom
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: {zoom_level}%;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Streamlit app
st.title("Real-Time Audio Classification")
st.markdown("### Predict whether the audio input is 'Singing' or 'Talking'!")

# Sidebar for uploading background image and zoom control
st.sidebar.header("Customize Background")
uploaded_bg_image = st.sidebar.file_uploader("Upload a background image", type=["jpg", "jpeg", "png"])

# Zoom control slider
zoom_level = st.sidebar.slider("Zoom Level", 50, 200, 100)  # Zoom level between 50% to 200%

# Apply the uploaded background image with zoom
add_background_image_from_upload(uploaded_bg_image, zoom_level)

# Initialize session state variables
if "is_classifying" not in st.session_state:
    st.session_state.is_classifying = False
if "last_predictions" not in st.session_state:
    st.session_state.last_predictions = []
if "confirmed_label" not in st.session_state:
    st.session_state.confirmed_label = None

# Function to preprocess audio
def preprocess_audio(audio_data):
    y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=60)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc, y

# Function to classify audio
def classify_audio(mfcc):
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(mfcc)
    confidence = prediction[0].item()  # Extract scalar confidence
    label = "Singing" if confidence > 0.99 else "Talking"
    return label, confidence

# Function to detect silence
def is_silent(audio_data, threshold=400):
    rms = np.sqrt(np.mean(audio_data ** 2))
    return rms < threshold

# Audio Classification Section
st.header("Audio Classification")
if st.button("Start Classification"):
    st.session_state.is_classifying = True

if st.button("Stop Classification"):
    st.session_state.is_classifying = False

if st.session_state.is_classifying:
    st.markdown("**Listening... Press 'Stop Classification' to end.**")
    try:
        while st.session_state.is_classifying:
            # Read audio
            audio_data = stream.read(22050)

            # Preprocess audio
            mfcc, y = preprocess_audio(audio_data)

            # Check for silence
            if is_silent(y):
                st.write("No sound detected.")
                continue

            # Classify audio
            label, confidence = classify_audio(mfcc)

            # Track recent predictions
            st.session_state.last_predictions.append(label)
            if len(st.session_state.last_predictions) > 4:
                st.session_state.last_predictions.pop(0)

            # Check if last 4 predictions are the same
            if len(set(st.session_state.last_predictions)) == 1:
                st.session_state.confirmed_label = st.session_state.last_predictions[0]

            # Display processing or the confirmed label
            if st.session_state.confirmed_label:
                st.write(f"Prediction: {st.session_state.confirmed_label}")
            else:
                st.write("Processing...")

            # Pause briefly to prevent overwhelming the UI
            time.sleep(0.5)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Gracefully close the stream if stopped
        if not st.session_state.is_classifying:
            stream.stop_stream()
            stream.close()
            p.terminate()


