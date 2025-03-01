import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set working directory and paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
class_indices_path = f"{working_dir}/class_indices.json"

# Load the pre-trained model and class names
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))


# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("Plant Disease Recognition System")
    home_image_path = "home_page.jpeg"  # Replace with your image path
    st.image(home_image_path, use_column_width=True)
    st.markdown(
        """
        Welcome to the Plant Disease Recognition System! 🌿🔍
        
        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.
        
        ### Why Choose Us?
        - **Accuracy:** Advanced machine learning techniques for precise detection.
        - **User-Friendly:** Simple and intuitive interface.
        - **Fast and Efficient:** Quick results for timely decisions.
        
        ### Get Started
        Use the sidebar to navigate to the **Disease Recognition** page and upload an image to start.
        """
    )

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown(
        """
        ### About the Dataset
        This dataset consists of approximately 1,63,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes.
        
        #### Content
        - **Train:** 43,456 images
        - **Validation:** 10,849 images
        - **Test:** 38 images
        
        The dataset is designed to assist in plant disease detection using machine learning algorithms.
        """
    )

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.title("Plant Disease Classifier")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img, caption="Uploaded Image")

        with col2:
            if st.button("Classify"):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f"Prediction: {prediction}")
