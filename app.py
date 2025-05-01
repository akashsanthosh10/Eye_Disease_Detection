from datetime import datetime
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

@st.cache_resource
def load_model_and_processor():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = AutoModelForImageClassification.from_pretrained("training_result\\resnet_version6")
    return processor, model

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): 
            return img 
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img


def preprocess_image(img, sigmaX=10):
    image = crop_image_from_gray(img)
    image = cv2.resize(image, (224, 224))   
    return image

def predict(image):
    image = preprocess_image(image)
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_id]
    probabilities = torch.softmax(logits, dim=-1)
    predicted_probability = probabilities[0, predicted_class_id].item()
    return predicted_label,predicted_probability



processor, model = load_model_and_processor()
model.eval()
label2id = {"cataract": 0, "diabetic_retinopathy": 1, "glaucoma": 2, "normal": 3}
id2label = {0: "cataract", 1: "diabetic_retinopathy", 2: "glaucoma", 3: "normal"}
model.config.label2id = label2id
model.config.id2label = id2label




# Custom styles for better aesthetics
st.markdown(
    """
    <style>
    .block-container {
        padding: 2rem;
    }
    .stButton > button {
        background-color: #007BFF;
        color: white;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        color: white;
    }
    .title {
        color:hsl(215, 100.00%, 50.00%);
        font-size: 50px;
        font-weight: bold;
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.markdown('<div class="title">MediScan: AI-Powered Medical Image Analysis</div>', unsafe_allow_html=True)

st.markdown(
    """
    **Welcome to MediScan**, an advanced AI tool for detecting eye diseases from medical images.

    Upload an image of your eye to identify conditions such as **Cataracts**, **Diabetic Retinopathy**, and **Glaucoma**.
    """,
    unsafe_allow_html=True
)


# Section: Patient Details
st.header("Patient Details")
# Input for Patient ID
patient_id = st.text_input("Enter Patient ID", placeholder="e.g., P12345")

# Dropdown for Gender
gender = st.selectbox("Select Gender", options=["Male", "Female", "Other"])

# Section: Eye Image Upload
st.header("Upload Eye Images")

# File uploader for left and right eye images
col1, col2 = st.columns(2)
with col1:
    left_eye_image = st.file_uploader("Upload Left Eye Image", type=["jpg", "jpeg", "png"], key="left_eye")
with col2:
    right_eye_image = st.file_uploader("Upload Right Eye Image", type=["jpg", "jpeg", "png"], key="right_eye")


# Display uploaded images side by side
if left_eye_image or right_eye_image:
    st.subheader("Uploaded Images")
    cols = st.columns(2)

    with cols[0]:
        if left_eye_image:
            st.image(left_eye_image, caption="Left Eye Image", use_container_width=True)
        else:
            st.write("Left Eye Image not uploaded")

    with cols[1]:
        if right_eye_image:
            st.image(right_eye_image, caption="Right Eye Image", use_container_width=True)
        else:
            st.write("Right Eye Image not uploaded")


 # Submit Button
if st.button("Run Analysis"):
    if not patient_id:
        st.warning("Please enter a valid Patient ID.")
    elif not left_eye_image and not right_eye_image:
        st.warning("Please upload at least one eye image.")
    else:
        st.success(f"Analysis started for Patient ID: {patient_id} ({gender}).")
        with st.spinner('Processing...'):
            # Initialize results dictionary
            results = {}
            if left_eye_image:
                image=Image.open(left_eye_image)
                image=np.array(image)
                predicted_label,predicted_probability=predict(image)
                results["Left Eye"] = (predicted_label, predicted_probability)
            if right_eye_image:
                image=Image.open(right_eye_image)
                image=np.array(image)
                predicted_label,predicted_probability=predict(image)
                results["Right Eye"] = (predicted_label, predicted_probability)
            st.subheader("Diagnosis Results")
             # Display Results
            for eye, (label, probability) in results.items():
                st.write(f"### {eye}")
                st.write(f"**Diagnosis:** {label}")
                st.write(f"**Confidence Level:** {probability * 100:.2f}%")
                if label != "Normal":
                    st.warning(f"Recommendation: Please consult an ophthalmologist for further evaluation of the {eye.lower()}.")
            
            
             # Summary
            st.header("Summary")
            summary_text = "### Findings\n"
            for eye, (label, probability) in results.items():
                summary_text += f"- **{eye}:** {label} ({probability * 100:.2f}%)\n"
            st.markdown(summary_text)


            report_content = f"""MediScan Analysis Report\n\n\nPatient ID: {patient_id}\nGender: {gender}\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nFindings:"""
            for eye, (label, probability) in results.items():
                report_content += f"\n- {eye}: {label} ({probability * 100:.2f}%)"
                if label != "normal":
                    report_content += f" - Recommendation: Consult an ophthalmologist."
            
            # Button to Download Report
            st.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"MediScan_Report_{patient_id}.txt",
                mime="text/plain"
            )
st.markdown("---")