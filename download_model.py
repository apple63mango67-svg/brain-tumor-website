"""
Download pre-trained brain tumor model from GitHub.
Run before first use: python download_model.py
"""
import os
import urllib.request
import ssl

# Reliable model sources (binary classification or 4-class we convert)
MODEL_SOURCES = [
    ("https://github.com/josephhu1/CNN-Brain-Tumor-MRI-Classification/raw/main/biohack-model.keras", "brain_tumor_model.keras"),
    ("https://github.com/vibhorjoshi/Brain-tumor-classification-MRI-scan/raw/main/keras_model.h5", "brain_tumor_model.h5"),
]
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MIN_MODEL_SIZE = 500_000  # Real models are typically several MB

def is_valid_model(path):
    """Check file exists and is a real model (not HTML/text)."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size < MIN_MODEL_SIZE:
        return False
    with open(path, "rb") as f:
        head = f.read(100)
    # Reject if starts with HTML/text
    if head.startswith(b"<!") or head.startswith(b"import ") or head.startswith(b"#"):
        return False
    return True

def download():
    os.makedirs(MODELS_DIR, exist_ok=True)
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    for url, filename in MODEL_SOURCES:
        path = os.path.join(MODELS_DIR, filename)
        print(f"Trying {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as resp:
                data = resp.read()
            if len(data) < MIN_MODEL_SIZE:
                print(f"  Skipped: file too small ({len(data)} bytes)")
                continue
            if data.startswith(b"<!") or data.startswith(b"import "):
                print(f"  Skipped: not a model file")
                continue
            with open(path, "wb") as f:
                f.write(data)
            if is_valid_model(path):
                print(f"Model saved to {path}")
                return True
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    print("Download failed. Run train_model.py to train your own model.")
    return False

if __name__ == "__main__":
    download()
