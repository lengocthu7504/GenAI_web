from base64 import b64encode
import os
import io
import numpy as np

from util.inference import annotate, predict
from PIL import Image, ImageOps
import requests
import streamlit as st

# Grounding DINO
from detect_DINO import detect, groundingdino_model, load_image
#sam
from sam import segment, draw_mask, sam_predictor
#mask create
from mask_create import create_mask

api_token = "PLXe_cIGtnKlSghKjtOoSrgFYK-YaySWE3Rsr9h-"
account_id = "f063ac09c48c645dcae9a6d4f478cc1b"

def image_to_int_array(image, format="PNG"):
    """Current Workers AI REST API consumes an array of unsigned 8 bit integers"""
    bytes = io.BytesIO()
    image.save(bytes, format=format)
    return list(bytes.getvalue())

st.title('Image detect+segment+inpaint')


image_upload = st.file_uploader("Upload a photo")
if image_upload is None:
    st.stop()

if image_upload is not None:
    img = Image.open(image_upload)
    original_size = img.size
    image_source, image = load_image(image_upload)


st.subheader('Image orginal')
st.image(image_upload)

prompt_chosse_object = st.text_input(label="Describe the object you want to change:")

#Detect object by prompt
if prompt_chosse_object != "":
    annotated_frame, detected_boxes = detect(image_source, image,text_prompt=prompt_chosse_object, model=groundingdino_model)
    # st.subheader('Result of detect')
    # st.image(annotated_frame, caption = prompt_chosse_object)
    #segmnet base on detect object
    if detected_boxes.nelement() !=0:
        segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
        annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)
        st.subheader('Result of segment')
        st.image(annotated_frame_with_mask)
        #creat mask
        image_source_pil, image_mask_pil, inverted_image_mask_pil = create_mask(image_source, segmented_frame_masks)
        # st.subheader('Result of mask')
        # st.image(image_mask_pil)
        # Inpainting form
        if image_mask_pil is not None:
            with st.form("Prompt"):
                prompt = st.text_input(label="What would you like to see replaced?")
                submitted = st.form_submit_button("Generate")
                if submitted:
                    model = "@cf/runwayml/stable-diffusion-v1-5-inpainting"
                    image_array = image_to_int_array(image_source_pil)
                    mask_array = image_to_int_array(image_mask_pil)
                    with st.spinner("Generating..."):
                        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
                        response = requests.post(
                            url,
                            headers={"Authorization": f"Bearer {api_token}"},
                            json={"prompt": prompt, "image": image_array, "mask": mask_array}
                        )
                        if response.ok:
                            generated_img = Image.open(io.BytesIO(response.content))
                            st.image(generated_img.resize(original_size), caption=prompt)
                        else:
                            st.error(f"Error {response.status_code}: {response.reason}")
                            st.code(f"Details: {response.text}")

