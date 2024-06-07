# Git project
``` bash
!git clone https://github.com/lengocthu7504/GenAI_web.git
%cd PIXEL_PILOT/create_web
```

# Set up for Change Background
```bash
!pip install controlnet_aux streamlit diffusers transformers accelerate safetensors
```

# Set up for Edit Image
```bash
!pip install -q -r requirements.txt

!git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
%cd /Grounded-Segment-Anything/GroundingDINO
!pip install -q .
%cd /Grounded-Segment-Anything/segment_anything
!pip install -q .
%cd ..
%cd ..
!rm -r Grounded-Segment-Anything
!rm -r PowerPaint-V1-stable-diffusion-inpainting

!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

# Run web streamlit
Copy IP and click on the link .lt then paste the IP  and click submit
``` bash
!wget -q -O - ipv4.icanhazip.com
!streamlit run create_web.py --server.enableXsrfProtection false &>/dev/null & npx localtunnel --port 8501
```
