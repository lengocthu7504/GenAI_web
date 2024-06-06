import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageChops
import numpy as np
import torch
import base64
from io import BytesIO
from functools import lru_cache

# Assuming these are defined in your setup
from iopaint.plugins.interactive_seg import InteractiveSeg, SEGMENT_ANYTHING_MODELS, RunPluginRequest

# Set up the device for model execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load and convert image from uploaded file
def load_image(image_file):
    img = Image.open(image_file).convert('RGB')
    return np.array(img)

# Function to encode image to base64
def encode_image_to_base64(image_file):
    """ Encodes an image file to base64 string, directly from binary data """
    return base64.b64encode(image_file.getvalue()).decode('utf-8')


@st.cache_resource
def load_model(model_name):
    """ Cache the model loading as it's expensive and doesn't need to be reloaded frequently """
    return InteractiveSeg(model_name=model_name, device=device)

def blend_image_with_mask(image, mask):
    # Convert mask to RGBA
    mask_colored = np.zeros((*mask.shape, 4), dtype=np.uint8)
    mask_colored[mask == 1] = [255, 255, 0, 128]  # Yellow color, half transparent
    mask_image = Image.fromarray(mask_colored, 'RGBA')
    
    # Overlay mask on original image
    return Image.alpha_composite(Image.fromarray(image).convert('RGBA'), mask_image)

# Streamlit UI
model_name = st.selectbox("Select Model", options=list(SEGMENT_ANYTHING_MODELS.keys()))
model = load_model(model_name)
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image_pil)
    image_base64 = encode_image_to_base64(uploaded_file)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=0,
        background_image=image_pil,
        update_streamlit=True,
        width=image_np.shape[1],
        height=image_np.shape[0],
        drawing_mode="point",
        point_display_radius=5,
        key="canvas",
    )

    if canvas_result.json_data and st.button('Generate Mask'):
        points = [
            [int(obj["left"]), int(obj["top"]), 1] for obj in canvas_result.json_data["objects"]
        ]
        # Create a RunPluginRequest object
        request = RunPluginRequest(
            name=model_name,
            image=image_base64,
            clicks=points
        )
        # mask = model.gen_mask(image_np, request)
        # blended_image = blend_image_with_mask(image_np, mask)
        # st.image(blended_image, caption='Output with Mask', use_column_width=True)
        
        mask = model.gen_mask(image_np, request)
        st.image(mask, caption='Generated Mask', use_column_width=True)
