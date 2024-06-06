import numpy as np
from PIL import Image
import torch

def create_mask(image_source, segmented_frame_masks):

    if isinstance(segmented_frame_masks[0][0], torch.Tensor):
        mask = segmented_frame_masks[0][0].cpu().numpy()
    else:
        mask = segmented_frame_masks 

    inverted_mask = ((1 - mask) * 255).astype(np.uint8)


    image_source_pil = Image.fromarray(image_source)
    image_mask_pil = Image.fromarray(mask)
    inverted_image_mask_pil = Image.fromarray(inverted_mask)


    return image_source_pil, image_mask_pil, inverted_image_mask_pil