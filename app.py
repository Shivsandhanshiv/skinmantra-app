import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
from fpdf import FPDF
import uuid

# ---------- Basic Config ----------
st.set_page_config(page_title="SkinMantra - AI Skin Cancer Detector", page_icon="🩺", layout="wide")
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .stButton>button {border-radius: 12px; padding: 0.5rem 1rem;}
    .stTabs [data-baseweb="tab"] { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource()
def load_model_cached():
   return load_model("E:/UMIT/MINI_project/implementation/skinmantra_model.keras")
model = load_model_cached()

# ---------- Label Encoder ----------
le = LabelEncoder()
le.classes_ = np.load('labels.npy', allow_pickle=True)

# ---------- Sidebar ----------
st.sidebar.image("assets/logo.png", width=200)
tabs = st.sidebar.radio("Navigate", ["🩺 Detection", "📂 History", "ℹ️ About"])

# ---------- Detection Page ----------
if tabs == "🩺 Detection":
    st.title("SkinMantra - AI based Skin Cancer Detector")
    col1, col2 = st.columns([2,2])

    with col1:
        st.subheader("Take a Real-Time Image")
        picture = st.camera_input("Capture Skin Image")

    with col2:
        st.subheader("Or Upload Images")
        uploaded_file = st.file_uploader("Drag & Drop or Browse", type=["jpg", "jpeg", "png"])

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif picture is not None:
        image = Image.open(picture)

    if image is not None:
        st.image(image, caption='Selected Image', use_container_width=True)

        # Preprocessing
        img = image.resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img)[0]
        predicted_class_idx = np.argmax(pred)
        predicted_class = le.inverse_transform([predicted_class_idx])[0]
        confidence = pred[predicted_class_idx] * 100

        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # Save History + PDF
        if st.button("Save Result & Generate PDF"):
            id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame([[id, timestamp, predicted_class, f"{confidence:.2f}%"]],
                              columns=['ID','Timestamp','Prediction','Confidence'])
            if os.path.exists('history.csv'):
                df.to_csv('history.csv', mode='a', header=False, index=False)
            else:
                df.to_csv('history.csv', index=False)

            # Generate PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="SkinMantra - AI Skin Cancer Report", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Date: {timestamp}", ln=True)
            pdf.cell(200, 10, txt=f"Prediction: {predicted_class}", ln=True)
            pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
            pdf.ln(5)
            pdf.multi_cell(0, 10, txt="Please consult a certified dermatologist for further diagnosis.")
            pdf.ln(10)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 8, txt="Thank you for reviewing our project. Your guidance has been invaluable to us. - Team SkinMantra")
            pdf.output(f"report_{id}.pdf")
            st.success("✅ PDF Report Generated")
            with open(f"report_{id}.pdf", "rb") as file:
                st.download_button("📄 Download Report", data=file, file_name="SkinMantra_Report.pdf", mime="application/pdf")

# ---------- History ----------
elif tabs == "📂 History":
    st.title("📂 Scan History")
    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        st.dataframe(df)
    else:
        st.warning("No scan history found.")

# ---------- About ----------
elif tabs == "ℹ️ About":
    st.title("About SkinMantra")
    st.write("""
        **SkinMantra** is an AI-powered application developed for early skin cancer detection using deep learning.

        ### 👩‍💻 Team Members:
        - Shivani Sandhanshiv
        - Riddhi Sawant
        - Pooja Tak

        ### 🌟 Features:
        - AI-Based Skin Cancer Prediction
        - Confidence Score
        - PDF Report Generator
        - Real-Time Camera Integration
        - Prediction History
        - Examiner Friendly "Thank You" Note

        ### ⚠️ Note:
        Always consult a certified medical professional before making health decisions.
    """)

# ---------- Footer ----------
st.markdown("<hr><center>SkinMantra - Powered by Deep Learning</center>", unsafe_allow_html=True)
