import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import random
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="Sampah Classifier", layout="wide")
st.title("🗑️ Trash Image Classifier (TFLite Version)")

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Contoh
IMG_SIZE = (224, 224)

# Load TFLite model
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

cnn_interpreter = load_tflite_model("cnn_model_final.tflite")
transfer_interpreter = load_tflite_model("trashnet_final.tflite")

def get_prediction(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction

def preprocess_input_mobilenet(x):
    x = x / 127.5 - 1.0
    return x

def augment_image(img):
    img = cv2.resize(img, IMG_SIZE)
    if st.session_state.get("flip", False):
        img = cv2.flip(img, 1)
    if st.session_state.get("zoom", False):
        zoom = random.uniform(1.0, 1.2)
        h, w = img.shape[:2]
        zh, zw = int(h / zoom), int(w / zoom)
        img = img[(h - zh)//2:(h + zh)//2, (w - zw)//2:(w + zw)//2]
        img = cv2.resize(img, (w, h))
    if st.session_state.get("brightness", False):
        img = np.clip(img * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    return img

with st.expander("📤 Upload Gambar Sampah"):
    uploaded_file = st.file_uploader("Unggah gambar sampah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    with st.expander("📷 Lihat Gambar yang Diupload"):
        st.image(image, caption="Gambar Asli", width=300)

    with st.expander("🛠️ Tahapan Preprocessing"):
        if st.button("1️⃣ Resize ke 224x224"):
            img_resized = cv2.resize(img_array, IMG_SIZE)
            st.image(img_resized, caption="Setelah Resize", channels="RGB", width=300)

        if st.button("2️⃣ Normalisasi (x / 127.5 - 1.0)"):
            img_resized = cv2.resize(img_array, IMG_SIZE)
            img_normalized = preprocess_input_mobilenet(img_resized.astype(np.float32))
            st.image((img_normalized + 1.0) / 2.0, caption="Setelah Normalisasi", channels="RGB", width=300)

        st.session_state.flip = st.checkbox("🔁 Horizontal Flip", value=False)
        st.session_state.zoom = st.checkbox("🔍 Zoom In", value=False)
        st.session_state.brightness = st.checkbox("💡 Brightness", value=False)

        if st.button("🔄 Terapkan Augmentasi Manual"):
            img_aug = augment_image(img_array)
            st.image(img_aug, caption="Hasil Augmentasi", width=300)

        if st.button("🔁 Reset Preprocessing"):
            st.session_state.flip = False
            st.session_state.zoom = False
            st.session_state.brightness = False
            st.rerun()

    with st.expander("🚀 Prediksi Gambar Sampah"):
        if st.button("🔮 Jalankan Prediksi"):
            img_resized = cv2.resize(img_array, IMG_SIZE)

            cnn_input = img_resized.astype(np.float32) / 255.0
            cnn_input = np.expand_dims(cnn_input, axis=0)

            transfer_input = preprocess_input_mobilenet(img_resized.astype(np.float32))
            transfer_input = np.expand_dims(transfer_input, axis=0)

            cnn_pred = get_prediction(cnn_interpreter, cnn_input)
            transfer_pred = get_prediction(transfer_interpreter, transfer_input)

            st.subheader("📊 Hasil Prediksi")
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Gambar yang Diupload", width=224)
                st.write("**Model CNN (TFLite):**")
                st.write(f"Prediksi: `{CLASS_NAMES[np.argmax(cnn_pred)]}`")
                st.bar_chart(cnn_pred)

            with col2:
                st.image(image, caption="Gambar yang Diupload", width=224)
                st.write("**Model Transfer Learning (TFLite):**")
                st.write(f"Prediksi: `{CLASS_NAMES[np.argmax(transfer_pred)]}`")
                st.bar_chart(transfer_pred)

else:
    st.info("Silakan unggah gambar terlebih dahulu.")
