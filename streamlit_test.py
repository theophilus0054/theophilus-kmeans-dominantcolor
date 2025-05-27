import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from io import BytesIO
import base64
import os
import traceback
import time

st.set_page_config(page_title="Image Clustering", page_icon=":guardsman:", layout="wide")

st.title("Image Clustering with KMeans")
st.write(
    "Try uploading an image to get major colors extracted from it using KMeans clustering. "
    "You can also adjust the number of clusters to see how it affects the results."
)

st.sidebar.header("Settings")
st.sidebar.write("Upload an image :gear:")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

MAX_IMAGE_SIZE = 2000  # 2000x2000 pixels

def convert_image_to_array(image):
    """
    Convert a PIL image to a numpy array.
    """
    return np.array(image)

# UI Layout
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Information about limitations
with st.sidebar.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Maximum file size: 10MB
    - Large images will be automatically resized
    - Supported formats: PNG, JPG, JPEG
    - Processing time depends on image size
    """)
    
if my_upload is not None:
    try:
        image = Image.open(my_upload)
        image = image.convert("RGB")  # pastikan dalam RGB
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if image.width > MAX_IMAGE_SIZE or image.height > MAX_IMAGE_SIZE:
            image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
            st.warning(f"Image resized to {image.size} for processing.")

        # Convert image to numpy array
        img_array = np.float32(convert_image_to_array(image)) / 255.0
        flat_img = img_array.reshape(-1, 3)

        # Jumlah cluster = 8 (default)
        n_clusters = 8

        with st.spinner("Clustering..."):
            km = KMeans(n_clusters=n_clusters, random_state=42)
            km.fit(flat_img)
            colors = (km.cluster_centers_ * 255).astype(np.uint8)

        # Ambil hanya 5 warna dominan teratas
        top_colors = colors[:5]  # atau gunakan slicing berdasarkan kebutuhan

        st.subheader("Dominant Colors (Top 5)")

        # Tampilkan masing-masing warna dalam kotak terpisah
        color_cols = st.columns(5)
        for idx, (col, color) in enumerate(zip(color_cols, top_colors)):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[:, :] = color  # Isi dengan warna dominan

            col.image(img, caption=f"Cluster {idx+1}", use_container_width=True)
            col.markdown(f"<p style='text-align:center;'>RGB: {tuple(color)}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Error processing image:")
        st.text(traceback.format_exc())
