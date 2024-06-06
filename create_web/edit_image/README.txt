#install dependence
%cd /content
!git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
!git clone https://huggingface.co/Sanster/PowerPaint-V1-stable-diffusion-inpaintingp;
%cd /content/Grounded-Segment-Anything
!pip install -q -r requirements.txt
%cd /content/Grounded-Segment-Anything/GroundingDINO
!pip install -q .
%cd /content/Grounded-Segment-Anything/segment_anything
!pip install -q .
%cd /content/Grounded-Segment-Anything
!pip install streamlit torch Pillow diffusers streamlit_drawable_canvas
!rm -r /content/Grounded-Segment-Anything
!rm -r /content/PowerPaint-V1-stable-diffusion-inpainting
%cd /content/final
! wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


#run
!wget -q -O - ipv4.icanhazip.com
!streamlit run web_app_ver2.py --server.enableXsrfProtection false &>/dev/null & npx localtunnel --port 8501


#reset gpu
import torch
torch.cuda.empty_cache()
