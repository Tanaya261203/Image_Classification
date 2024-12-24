import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Load the trained model
model = load_model('imgclassification.h5')

# Define the labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Set Streamlit page configuration
st.set_page_config(page_title="Futuristic Image Classifier", page_icon="🖼️", layout="wide")

# Custom CSS for advanced futuristic design and black background
st.markdown("""
    <style>
        body {
            background: #000000;  /* Black background */
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
            padding: 0;
            margin: 0;
            overflow-x: hidden;
        }

        .stButton button {
            background: linear-gradient(135deg, #ff6a00, #ee0979);
            border: none;
            padding: 18px 40px;
            font-size: 1.5rem;
            border-radius: 25px;
            color: #fff;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(255, 105, 180, 0.5), 0 0 15px rgba(255, 105, 180, 0.7);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }

        .stButton button:hover {
            transform: scale(1.2);
            box-shadow: 0 0 20px rgba(255, 105, 180, 0.5), 0 0 30px rgba(255, 105, 180, 0.8);
        }

        h1 {
            font-size: 4rem;
            text-align: center;
            color: #ff6a00;
            text-shadow: 0 0 20px #ff6a00, 0 0 30px #ff6a00, 0 0 40px #ee0979;
            animation: neon-glow 1.5s ease-in-out infinite alternate;
        }

        h2 {
            font-size: 2rem;
            color: #ffffff;
            text-align: center;
            margin-top: 20px;
        }

        .main-container {
            text-align: center;
            position: relative;
        }

        .upload-section {
            margin-top: 40px;
        }

        .result-img {
            max-width: 80%;  /* Limit width to 80% */
            height: auto;
            object-fit: cover;
            border-radius: 15px;
            border: 2px solid #ff6a00;
            box-shadow: 0 0 25px rgba(255, 105, 180, 0.5), 0 0 50px rgba(255, 105, 180, 0.8);
            margin-top: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .result-img:hover {
            transform: scale(1.05);
            box-shadow: 0 0 35px rgba(255, 105, 180, 0.5), 0 0 70px rgba(255, 105, 180, 0.8);
        }

        .prediction {
            font-size: 2rem;
            color: #ff6a00;
            font-weight: bold;
            margin-top: 20px;
        }

        .top-prediction {
            font-size: 2.5rem;  /* Increased font size for top predictions */
            color: #ffffff;
            font-weight: bold;
            margin-top: 10px;
        }

        @keyframes neon-glow {
            0% {
                text-shadow: 0 0 10px #ff6a00, 0 0 20px #ff6a00, 0 0 30px #ff6a00;
            }
            50% {
                text-shadow: 0 0 30px #ff6a00, 0 0 40px #ff6a00, 0 0 50px #ee0979;
            }
            100% {
                text-shadow: 0 0 10px #ff6a00, 0 0 20px #ff6a00, 0 0 30px #ff6a00;
            }
        }

        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }

        .particle-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle, rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.9));
            pointer-events: none;
            z-index: -1;
            animation: particles 2s infinite;
        }

        @keyframes particles {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.1);
            }
            100% {
                transform: scale(1);
            }
        }
    </style>
""", unsafe_allow_html=True)

# Add background particles
st.markdown('<div class="particle-background"></div>', unsafe_allow_html=True)

# Header Section
st.title("🌟 Galactic Image Classifier")
st.write("Upload your image and discover its galactic identity!")

# Create Upload Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image using PIL
    image = Image.open(uploaded_file)

    # Preprocess the image
    image = image.convert("RGB")  # Ensure the image is RGB
    image = image.resize((32, 32))  # Resize to match model input size
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Display the result in a glowing box with interactive animation
    st.markdown(f'<div class="prediction">Classifying your image...</div>', unsafe_allow_html=True)

    # Predict the class using the trained model
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    # Add a slight delay for suspense
    time.sleep(1)

    # Display the uploaded image with size adjustments
    st.image(image, caption="Uploaded Image", use_column_width='auto', width=600, channels="RGB")

    # Display the predicted class with the prediction probability
    predicted_prob = np.max(prediction) * 100
    st.markdown(f'<div class="prediction">Predicted Class: {predicted_class}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction">Prediction Probability: {predicted_prob:.2f}%</div>', unsafe_allow_html=True)

    # Display the top 3 predicted classes with probabilities and increased font size
    st.markdown("<h2>Top 3 Predictions:</h2>", unsafe_allow_html=True)
    top_3_indices = np.argsort(prediction[0])[-3:][::-1]
    for i in top_3_indices:
        st.markdown(f'<div class="top-prediction">{labels[i]}: {prediction[0][i] * 100:.2f}%</div>', unsafe_allow_html=True)
