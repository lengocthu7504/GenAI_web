# Set up for Change Background

```bash
!pip install controlnet_aux streamlit diffusers transformers accelerate safetensors
```

# Set up for Edit Image

```bash
%cd /content/drive/MyDrive/PIXEL_PILOT/create_web
!pip install -q -r requirements.txt
!pip install -q .
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

# Run web streamlit
Copy IP and click on the link .lt then paster the IP  and  submit
``` bash
!wget -q -O - ipv4.icanhazip.com
!streamlit run create_web.py --server.enableXsrfProtection false &>/dev/null & npx localtunnel --port 8501
```
