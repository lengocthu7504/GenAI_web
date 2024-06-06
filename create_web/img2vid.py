import os, math, torch, cv2
import gradio as gr
import random

from omegaconf import OmegaConf
from glob import glob
from pathlib import Path
from typing import Optional
import numpy as np
from einops import rearrange, repeat

from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as TF
from sgm.util import instantiate_from_config

def load_model(config: str, device: str, num_frames: int, num_steps: int):
    config = OmegaConf.load(config)
    config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = device
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (num_frames)
    with torch.device(device):
        model = instantiate_from_config(config.model).to(device).eval().requires_grad_(False)
    return model

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N, T, device, dtype=None):
    batch = {}
    batch_uc = {}
    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device, dtype=dtype),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]
    if T is not None:
        batch["num_video_frames"] = T
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def sample(
    model,
    input_image,
    resize_image: bool = False,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = "/content/outputs",
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    torch.manual_seed(seed)

    
    all_img_paths = [input_image]
    
    all_out_paths = []
    for input_img_path in all_img_paths:
        with Image.open(input_img_path) as image:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            if resize_image and image.size != (1024, 576):
                print(f"Resizing {image.size} to (1024, 576)")
                image = TF.resize(TF.resize(image, 1024), (576, 1024))
            w, h = image.size
            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                image = image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )
            image = ToTensor()(image)
            image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )
        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")
        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug
        # low vram mode
        import gc
        gc.collect()
        model.conditioner.cpu()
        model.first_stage_model.cpu()
        torch.cuda.empty_cache()
        model.sampler.verbose = True

        with torch.no_grad():
            with torch.autocast(device):
                model.conditioner.to(device)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )
                model.conditioner.cpu()
                torch.cuda.empty_cache()

                # from here, dtype is fp16
                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)
                for k in uc.keys():
                    uc[k] = uc[k].to(dtype=torch.float16)
                    c[k] = c[k].to(dtype=torch.float16)

                randn = torch.randn(shape, device=device, dtype=torch.float16)
                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(2, num_frames).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                for k in additional_model_inputs:
                    if isinstance(additional_model_inputs[k], torch.Tensor):
                        additional_model_inputs[k] = additional_model_inputs[k].to(dtype=torch.float16)

                def denoiser(input, sigma, c):
                    return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                samples_z.to(dtype=model.first_stage_model.dtype)
                model.en_and_decode_n_samples_a_time = decoding_t
                model.first_stage_model.to(device)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                model.first_stage_model.cpu()
                torch.cuda.empty_cache()

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    fps_id + 1,
                    (samples.shape[-1], samples.shape[-2]),
                )
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                for frame in vid:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)
                writer.release()
                all_out_paths.append(video_path)
    return all_out_paths

def infer(model, input_path: str, resize_image: bool, n_frames: int, n_steps: int) -> str:
  
  seed = random.randint(0, 2**32)
  seed = int(seed)
  print('Seed of this video is: %d'%seed)
  output_paths = sample(
    model,
    input_image=input_path,
    resize_image=resize_image,
    num_frames=n_frames,
    num_steps=n_steps,
    
    motion_bucket_id=127,
    cond_aug=0.02,
    seed=seed,
    device='cuda'
  )
  return output_paths[0]

# def resize_image(uploaded_image, resize_option, crop_offset, resized_image_file):
#   '''
#   Resize the input input based on the resize option.
#   Adjust the crop position based on the crop offset
#   '''

#   if uploaded_image is None:
#     return None

#   w, h = (1024, 576) # target width and height
#   old_h, old_w = uploaded_image.shape[0:2]


#   # Crop image
#   if resize_option == "Just resize":
#     cropped_image = uploaded_image
#   elif resize_option == "Crop and resize":
#     # crop to same aspect ratio
#     if w/h > old_w/old_h:
#       target_h = int(old_w * h/w)
#       padding1 = int((old_h - target_h)/2)
#       padding2 = old_h -padding1
#       padding = int(np.interp(crop_offset, [0, 0.5, 1], [0, padding1, padding1 + padding2]))
#       cropped_image = uploaded_image[padding:padding+target_h, :, :]
#     else:
#       target_w = int(old_h * w/h)
#       padding1 = int((old_w - target_w)/2)
#       padding2 = old_w - padding1
#       padding = int(np.interp(crop_offset, [0, 0.5, 1], [0, padding1, padding1 + padding2]))
#       cropped_image = uploaded_image[:, padding:padding+target_w, :]
#   else:
#     ValueError(f"Unknown resize option: {resize_option}")


#   # Resize image
#   resized_image_file = "/content/resized_image.png"  # cannot be jpg for some reason
#   resized_image = cv2.resize(cropped_image, (w, h))
#   im = Image.fromarray(resized_image)
#   im.save(resized_image_file)
#   return resized_image,resized_image_file


def image2video(image):
  num_frames = 25
  num_steps = 30
  model_config = "generative-models/scripts/sampling/configs/svd_xt.yaml"
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = load_model(model_config, device, num_frames, num_steps)
  model.conditioner.cpu()
  model.first_stage_model.cpu()
  model.model.to(dtype=torch.float16)
  torch.cuda.empty_cache()
  model = model.requires_grad_(False)

  resize_choices = ["Just resize", "Crop and resize"]
  resize_default_index = 1

  vid = infer(model, image, resize_image=False, n_frames=num_frames, n_steps=num_steps)
  return vid
