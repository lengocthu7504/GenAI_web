import torch
from PIL import Image, ImageOps
import numpy as np
import cv2

# Function to add tasks to prompt
def add_task_to_prompt(prompt, negative_prompt, task):
    if task == "object-removal":
        promptA = prompt + " P_ctxt"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"
    elif task == "shape-guided":
        promptA = prompt + " P_shape"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
    elif task == "image-outpainting":
        promptA = prompt + " P_ctxt"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"
    else:
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
    return promptA, promptB, negative_promptA, negative_promptB

# Main processing function
@torch.inference_mode()
def predict(
    pipe,
    input_image,
    prompt,
    fitting_degree,
    ddim_steps,
    scale,
    negative_prompt,
    task,
):
    # Adjusting image size based on aspect ratio
    width, height = input_image["image"].convert("RGB").size
    if width < height:
        input_image["image"] = (
            input_image["image"].convert("RGB").resize((640, int(height / width * 640)))
        )
    else:
        input_image["image"] = (
            input_image["image"].convert("RGB").resize((int(width / height * 640), 640))
        )

    promptA, promptB, negative_promptA, negative_promptB = add_task_to_prompt(
        prompt, negative_prompt, task
    )
    img = np.array(input_image["image"].convert("RGB"))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image["image"] = input_image["image"].resize((H, W))
    input_image["mask"] = input_image["mask"].resize((H, W))

    result = pipe(
        promptA=promptA,
        promptB=promptB,
        tradoff=fitting_degree,
        tradoff_nag=fitting_degree,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image["image"].convert("RGB"),
        mask_image=input_image["mask"].convert("RGB"),
        guidance_scale=scale,
        num_inference_steps=ddim_steps,
    ).images[0]
    return result

# Load the model
def gen_image(pipe, uploaded_image, uploaded_mask, prompt, negative_prompt, task):
    pipe = pipe
    image = uploaded_image
    mask = uploaded_mask
    input_image = {"image": image, "mask": mask}

    tasks = [
    {
        "task": "object-removal",
        "guidance_scale": 12,
        "prompt": "",
        "negative_prompt": "",
    },
    {
        "task": "shape-guided",
        "guidance_scale": 7.5,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    },
    {
        "task": "inpaint",
        "guidance_scale": 7.5,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    },
    {
        "task": "image-outpainting",
        "guidance_scale": 7.5,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    },
]
    
    if task == "image-outpainting":
            margin = 128
            input_image["image"] = ImageOps.expand(
                input_image["image"],
                border=(margin, margin, margin, margin),
                fill=(127, 127, 127),
            )
            outpaint_mask = np.zeros_like(np.asarray(input_image["mask"]))
            input_image["mask"] = Image.fromarray(
                cv2.copyMakeBorder(
                    outpaint_mask,
                    margin,
                    margin,
                    margin,
                    margin,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
            )
    for i in tasks:
         if i["task"] == task:
            guidance_scale = i["guidance_scale"]
            prompt = i["prompt"]
            negative_prompt = i["negative_prompt"]
            result_image = predict(
                pipe,
                input_image,
                prompt,
                1,  # fitting_degree
                30,  # ddim_steps
                guidance_scale,
                negative_prompt,
                task,
            )
            return result_image