import requests
import os

def download_model(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
    except requests.RequestException as e:
        print(f"Failed to download {filename}. Error: {e}")

def download_models():
    urls = {
        "fcpe.pt": "https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt",
        "hubert_base.pt": "https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/hubert_base.pt",
        "rmvpe.pt": "https://huggingface.co/Kit-Lemonfoot/RVC_DidntAsk/resolve/main/rmvpe.pt"
    }
    
    for filename, url in urls.items():
        if not os.path.exists(filename):
            download_model(url, filename)
