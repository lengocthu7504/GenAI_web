import edit_image.datasets.transforms as T
from edit_image.models import build_model
from edit_image.util.slconfig import SLConfig
from edit_image.util.utils import clean_state_dict
from edit_image.util.inference import annotate, predict
import supervision as sv
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_file):
    """Load and transform an image file for detection and visualization."""
    if image_file is None:
        return None, None
    image = Image.open(image_file).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_transformed, _ = transform(image, None)
    image_array = np.array(image)
    return image_array, image_transformed

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"


groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

# detect object using grounding DINO
def detect(image_source, image, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )

  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB
  return annotated_frame, boxes

# annotated_frame, detected_boxes = detect(image, text_prompt="bench", model=groundingdino_model)
# Image.fromarray(annotated_frame)