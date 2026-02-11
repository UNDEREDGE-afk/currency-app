import gdown
import os

if not os.path.exists("my_model.keras"):
    url = "https://drive.google.com/uc?id=1hDJOiAQIwN-TpoPQVai5Wu1EcIPkr14S"
    gdown.download(url, "my_model.keras", quiet=False)
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Define class labels
class_names = {0: '100', 1: '500'}

# Load the trained model (do this once to avoid reloading)
@st.cache_resource
def load_my_model():
    return load_model("my_model.keras")

model = load_my_model()

st.set_page_config(page_title="Curency Note Classifier", page_icon="🖼️", layout="centered")

st.title("Curency Note Classifier")
st.write("Capture to Find the Value of the Currency")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names.get(predicted_class_index, "Unknown")

    st.success(f"✅ Predicted Class: **{predicted_class}**")
