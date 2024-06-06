import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import streamlit as st
import torch
import time

# Import custom modules
from edit_image.detect_DINO import detect, groundingdino_model, load_image
from edit_image.sam import segment, draw_mask, sam_predictor
from edit_image.mask_create import create_mask
from edit_image.powerpaint import gen_image
from edit_image.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from edit_image.power_paint_tokenizer import PowerPaintTokenizer

def robust_load_model(retry_limit=3, backoff_factor=2):
    attempts = 0
    while attempts < retry_limit:
        try:
            pipe = Pipeline.from_pretrained(
                "Sanster/PowerPaint-V1-stable-diffusion-inpainting",
                torch_dtype=torch.float16,
                safety_checker=None,
                variant="fp16",
            )
            pipe.tokenizer = PowerPaintTokenizer(pipe.tokenizer)
            if torch.cuda.is_available():
                return pipe.to("cuda")
            else:
                return pipe.to("cpu")
        except Exception as e:
            attempts += 1
            wait_time = backoff_factor ** attempts
            st.error(f"Failed to load model on attempt {attempts}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    st.error("Failed to load model after several attempts. Please check your connection and try again later.")
    return None

@st.cache_resource
def load_model_gen():
    return robust_load_model()

pipe = load_model_gen()
if pipe is None:
    st.stop()

image_upload = st.file_uploader("Upload a photo")
task_options = ('object-removal', 'shape-guided', 'inpaint', 'image-outpainting')
mask_creation_methods = ('Use Prompt (best for remove)', 'Draw Mask')

current_task = st.radio("Choose task:", task_options)
current_mask_creation_method = st.radio("Choose the method to create a mask:", mask_creation_methods)

# Reset mask and related states when the method is changed or new image is uploaded
if 'prev_mask_method' not in st.session_state or st.session_state.prev_mask_method != current_mask_creation_method:
    st.session_state.image_mask_pil = None
    st.session_state.prev_mask_method = current_mask_creation_method

if 'prev_image' not in st.session_state or st.session_state.prev_image != image_upload:
    st.session_state.image_mask_pil = None
    st.session_state.prev_image = image_upload

if image_upload is None:
    st.stop()

if image_upload is not None:
    st.session_state.image_source, image = load_image(image_upload)
    if current_mask_creation_method == 'Use Prompt (best for remove)':
        st.subheader('Image Original')
        st.image(st.session_state.image_source)
        prompt_choose_object = st.text_input("Describe the object you want to segment:", key="prompt_object")
        if prompt_choose_object:
            annotated_frame, detected_boxes = detect(st.session_state.image_source, image, text_prompt=prompt_choose_object, model=groundingdino_model)
            if detected_boxes.nelement() != 0:
                segmented_frame_masks = segment(st.session_state.image_source, sam_predictor, boxes=detected_boxes)
                annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)
                st.subheader('Result of Segment')
                st.image(annotated_frame_with_mask)
                try:
                    st.session_state.image_source_pil, st.session_state.image_mask_pil, _ = create_mask(st.session_state.image_source, segmented_frame_masks)
                    if st.session_state.image_mask_pil is not None:
                        st.subheader('Result of Mask')
                        st.image(st.session_state.image_mask_pil)
                    else:
                        st.error("Mask creation failed.")
                except Exception as e:
                    st.error(f"Error in mask creation: {e}")
            else:
                st.warning("No objects detected. Please try a different prompt or image.")
        

    elif current_mask_creation_method == 'Draw Mask':
        st.subheader('Draw on the image based on the selected task')
        stroke_width = st.slider("Stroke width: ", 1, 25, 5)
        h, w = st.session_state.image_source.shape[:2]
        scale_factor = 800 / max(w, h) if max(w, h) > 800 else 1
        w_, h_ = int(w * scale_factor), int(h * scale_factor)
        canvas_result = st_canvas(
            fill_color='rgba(255, 255, 255, 0)', stroke_width=stroke_width, stroke_color='white',
            background_image=Image.fromarray(st.session_state.image_source).resize((w_, h_)),
            update_streamlit=True, height=h_, width=w_, drawing_mode='freedraw', key="canvas"
        )
        if canvas_result.image_data is not None:
            mask = cv2.cvtColor(np.array(canvas_result.image_data), cv2.COLOR_RGBA2GRAY)
            mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            try:
                st.session_state.image_source_pil, st.session_state.image_mask_pil, _ = create_mask(st.session_state.image_source, mask)
                if st.session_state.image_mask_pil is not None:
                    st.subheader('Result of Mask')
                    st.image(st.session_state.image_mask_pil, caption='Generated Mask')
                else:
                    st.error("Mask creation failed.")
            except Exception as e:
                st.error(f"Error in mask processing: {e}")

if st.session_state.image_mask_pil is not None:
    with st.form("Prompt"):
        prompt_label = "Describe the change you want:" if current_task != "image-outpainting" else "Describe the outpainting you want:"
        st.session_state.prompt = st.text_input(label=prompt_label)
        negative_prompt = "out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature"
        submitted = st.form_submit_button("Generate")
        if submitted:
            result_image = gen_image(pipe, st.session_state.image_source_pil, st.session_state.image_mask_pil, st.session_state.prompt, negative_prompt, current_task)
            st.image(result_image, caption="Processed Image")
            dowdload_image = st.form_submit_button("download")
            
            if dowdload_image:
                st.experimental_rerun()