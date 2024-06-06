import PIL
from PIL import Image
import torch
# from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from controlnet_aux import PidiNetDetector, HEDdetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


def model(image, prompt, num):
    # model_id = "timbrooks/instruct-pix2pix"
    # pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    # pipe.to("cuda")
    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    #
    #
    # image = Image.open(image)
    # images = pipe(
    # prompt,
    # image=image,
    # num_inference_steps=10,
    # image_guidance_scale= 1,
    # num_images_per_prompt=num
    # ).images

    checkpoint = "lllyasviel/control_v11p_sd15_softedge"
    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(image, safe=True)


    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images
    return image
