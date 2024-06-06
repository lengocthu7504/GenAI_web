import io
import base64
from PIL import Image
import streamlit as st
import numpy as np

def get_image_download_link(image, filename='download.jpeg', text='Download Image', original_size=None):
    """Generates a link to download a particular image file."""
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8))  # Đảm bảo rằng dữ liệu là uint8 để từ ndarray chuyển sang image
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError("Unsupported image type. `image` must be either a numpy array or PIL.Image.")
    if original_size:
        img = img.resize(original_size, Image.LANCZOS)  # Sử dụng phương pháp LANCZOS cho việc resize chất lượng cao
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')  # Sử dụng định dạng JPEG cho việc lưu
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    return href
