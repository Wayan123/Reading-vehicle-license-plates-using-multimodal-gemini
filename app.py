import streamlit as st
import base64
import os
import pandas as pd
from PIL import Image
from io import BytesIO
import re
import json
import requests
import tempfile
import cv2

import google.generativeai as genai

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set Google API key 
genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

# Create the Model
txt_model = genai.GenerativeModel('gemini-pro')
vis_model = genai.GenerativeModel('gemini-pro-vision')

# Load data from CSV file
data = pd.read_csv('daftar_plat_kendaraan_indonesia.csv', encoding='ISO-8859-1')

# Image to Base 64 Converter
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Function to get LLM response
def get_llm_response(text, img):
    if not img:
        response = txt_model.generate_content(text)
    else:
        img = Image.open(img)
        response = vis_model.generate_content([text, img])
    
    # Mengambil semua kombinasi huruf kapital yang berdiri sendiri sebagai kata dalam teks menggunakan ekspresi reguler
    plat_codes = re.findall(r'\b(?:BA|BB|BK|BD|BE|BG|BH|BL|BM|BN|BP|A|B|D|E|F|T|Z|G|H|K|R|AA|AB|AD|L|M|N|P|S|W|AE|AG|DH|DK|DR|EA|EB|ED|DA|KB|KH|KT|KU|DB|DC|DD|DL|DM|DN|DP|DT|DW|DE|DG|PA|PB)\b', response.text)
    
    return response.text, plat_codes

# Function to capture image from webcam
def capture_webcam_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return frame

# Streamlit UI
def main():
    st.title("Detect vehicle license plate numbers and their region codes in Indonesia using Multimodal LLMs")

    col1_width = 0.5  # Lebar kolom kiri (50% dari layar)
    col2_width = 1.0 - col1_width  # Lebar kolom kanan (sisa dari layar)

    uploaded_img = None  # Inisialisasi uploaded_img di awal

    with st.sidebar:  # Panel kiri
        st.header("Uploaded Vehicle License Plate Numbers Image")
        option = st.radio("Choose Image Source:", ("Local Upload", "Link")) #, "Webcam")
        
        if option == "Link":
            image_url = st.text_input("Enter Image URL:")
            if image_url:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                uploaded_img = BytesIO()
                img.save(uploaded_img, format='JPEG')
                uploaded_img.seek(0)
                st.image(img, caption='Uploaded  Image', use_column_width=True)
        elif option == "Local Upload":
            uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg"])
            if uploaded_img is not None:
                st.image(uploaded_img, caption='Uploaded Image', use_column_width=True)
        
        # elif option == "Webcam":
        #     if st.button("Take Picture"):
        #         webcam_img = capture_webcam_image()
        #         uploaded_img = Image.fromarray(webcam_img)
         #       st.image(uploaded_img, caption='Webcam Image', use_column_width=True)
        
        
    with st.expander("Instructions"):  # Petunjuk di panel kiri
        st.write("Upload an image or type your message in the textbox.")
        st.write("Then click 'Submit' to see the AI response.")
        st.write("If an =IndexError: list index out of range= error occurs, try submit again or replace the image with a better quality.")
        st.write("Contoh prompt untuk 1 gambar: Berapa nomor plat kendaraan dan tahun aktifnya jika ada?")
        st.write("Contoh prompt jika lebih dari 1 gambar: Identifikasikan semua nomor plat kendaraan dan tahun aktif semuanya jika ada?")

    user_input_text = st.text_input("Masukan yang ingin anda tanyakan:", "Berapa nomor plat kendaraan dan tahun aktifnya jika ada?")

    if st.button("Submit"):
        if user_input_text:
            if uploaded_img is not None:
                st.image(uploaded_img, use_column_width=True)
                response_text, plat_codes = get_llm_response(user_input_text, uploaded_img)
                st.write("AI: ", response_text)
                
                # Mencocokkan setiap kode plat dengan informasi dalam data CSV
                for code in plat_codes:
                    match = data[data['Kode'] == code]
                    if not match.empty:
                        st.write("Kode Plat:", code)
                        st.write("Wilayah:", match['Wilayah'].iloc[0])
                        st.write("Kota:", match['Kota'].iloc[0])
                    else:
                        st.write("Kode Plat:", code)
                        st.write("Informasi Wilayah Tidak Ditemukan")
            else:
                st.error("Please upload an image or select webcam option.")

if __name__ == "__main__":
    main()
