import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps  
import numpy as np
import streamlit as st 
from dotenv import load_dotenv 
import os
import h5py
from io import BytesIO  # Import BytesIO from the io module

# Load the environment variables
load_dotenv()

# Load the GEMINI API key
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Open and modify the model configuration
f = h5py.File("keras_model3.h5", mode="r+")

# Ensure that model_config_string is not None before using methods on it
model_config_string = f.attrs.get("model_config")
if model_config_string and model_config_string.find('"groups": 1,') != -1:
    model_config_string = model_config_string.replace('"groups": 1,', '')
    f.attrs.modify('model_config', model_config_string)
    f.flush()

f.close()

def classify_waste(img):
    np.set_printoptions(suppress=True)
    try:
        model = load_model("keras_model3.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    # Load the labels
    try:
        class_names = open("labels3.txt", "r").readlines()
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return None, None

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Ensure the image is correctly processed
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name.strip(), confidence_score

# Set Streamlit page configuration
st.set_page_config(layout='wide')
st.title("E-Waste Management")

# Selection for input method
st.info("Choose an input method:")
# st.markdown("### Choose an input method:")

input_method = st.radio("", ("Webcam", "Upload Image"))

# Webcam or file uploader based on selection
input_img = None
if input_method == "Webcam":
    input_img = st.camera_input("Take a picture")
elif input_method == "Upload Image":
    input_img = st.file_uploader("Enter your image", type=['jpg', 'png', 'jpeg'])

# Process the image if available
if input_img is not None:
    if st.button("Classify"):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.info("Your uploaded Image")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Your Result")
            # Handle image processing based on input source
            image_file = Image.open(BytesIO(input_img.read())) if isinstance(input_img, BytesIO) else Image.open(input_img)
            label, confidence_score = classify_waste(image_file)
            if label and confidence_score:
                # Display classification result
                sdg_images = {
                    "0 battery": ["sdg goals/3.png", "sdg goals/6.jpg", "sdg goals/12.png", "sdg goals/13.png"],
                    "1 cardboard": ["sdg goals/12.png"],
                    "2 glass": ["sdg goals/12.png"],
                    "3 leather": ["sdg goals/12.png"],
                    "4 medical": ["sdg goals/3.png", "sdg goals/6.jpg", "sdg goals/12.png", "sdg goals/14.png", "sdg goals/15.png"],
                    "5 metal": ["sdg goals/3.png", "sdg goals/6.jpg", "sdg goals/12.png", "sdg goals/14.png"],
                    "6 plastic": ["sdg goals/3.png", "sdg goals/6.jpg", "sdg goals/12.png", "sdg goals/14.png", "sdg goals/15.png"],
                    "7 wood": ["sdg goals/12.png", "sdg goals/15.png"]
                }
                waste_type = label.split()[1].strip()

                st.success(f"The image is classified as {waste_type.upper()} waste.")

                col4, col5 = st.columns([1, 1])
                images = sdg_images.get(label)
                for i, image in enumerate(images):
                    if i % 2 == 0:
                        with col4:
                            st.image(image, use_column_width=True)
                    else:
                        with col5:
                            st.image(image, use_column_width=True)

        with col3:
            # Display recycle or not message and image
            recyclable_waste = ["0 battery", "1 cardboard", "5 metal", "6 plastic"]
            if label in recyclable_waste:
                st.success(f"This waste, such as {waste_type}, can be recycled.")
                with st.columns([1])[0]:
                    st.image("bin_images/recycle.gif", use_column_width=True)
            else:
                st.error(f"This waste, such as {waste_type}, can't be recycled.")
                with st.columns([1])[0]:
                    st.image("bin_images/non_recycle.gif", use_column_width=True)

        # Tutorial section for specific labels
        if label == "0 battery":
            st.info("Tutorial for BATTERY disposal")
            st.video("videos/battery.mp4", format="video/mp4", start_time=0, loop=False, autoplay=False, muted=False)
        elif label == "1 cardboard":
            st.info("Tutorial for CARDBOARD disposal")
            st.video("videos/cardboard.mp4", format="video/mp4", start_time=0, loop=False, autoplay=False, muted=False)
