import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the Inception model
model = load_model('/content/drive/MyDrive/trained_model.h5')

label_dict = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'normal', 3:'pituitary_tumor' } 

# Helper function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))    
    img = img_to_array(img)      
    img = preprocess_input(img)
    img = np.array(img, dtype="float32") 
    return img

# Helper function to make predictions on the image
def predict_image(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    i = predictions.argmax(axis=1)[0]
    predicted_label = label_dict[i]    
    return predicted_label

# Streamlit app
def main():
    st.title("Image Classifier with Inception Model")
    st.write("Upload an image and the model will predict its class.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make predictions on the image
        predictions = predict_image(image)

        # Display the predictions
        st.write("Predictions:")
        for _, label, confidence in predictions:
            st.write(f"- {label}: {confidence:.4f}")

if __name__ == "__main__":
    main()

# python -m streamlit run app.py --server.port 8080    