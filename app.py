import streamlit as st
import os
import pandas as pd
from PIL import Image
import zipfile
from io import BytesIO

from lib.prediction_pipeline import ensemble_predict

# Function to process images in the selected directory
def process_images(file_list):
    # Your image processing logic here
    # Replace the following example code with your actual processing logic
    result = []
    for uploaded_file in file_list:
        img_path = BytesIO(uploaded_file.read())
        # Replace this with your actual function that processes an image
        processed_result = ensemble_predict(img_path)
        result.append(processed_result)
    return result

# Function to generate and download a zip file
def download_zip(result):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for item in result:
            filename = item['filename']
            class_identified = item['class']
            confidence = item['confidence']
            content = f"{filename}\nClass: {class_identified}\nConfidence: {confidence}\n"
            zip_file.writestr(filename.replace('.jpg', '_result.txt'), content)
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit app
def main():
    st.title("EcoSentry App")

    # Upload multiple files
    uploaded_files = st.file_uploader("Choose multiple files:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Process images on button click
    if st.button("Process Images"):
        if uploaded_files:
            result = process_images(uploaded_files)
            st.success("Image processing completed!")

            # Download results as a zip file
            zip_buffer = download_zip(result)
            st.markdown(
                f"### [Download Results as Zip File](data:application/zip;base64,{zip_buffer.read().encode('base64')})"
            )

            # Display results
            st.header("Image Processing Results")
            for item in result:
                st.write(f"Filename: {item['filename']}")
                st.write(f"Class Identified: {item['class']}")
                st.write(f"Confidence: {item['confidence']}")
                st.write("---")

# Run the Streamlit app
if __name__ == "__main__":
    main()