"""
Download pre-trained brain tumor model at runtime.
Creates models/ if needed; downloads brain_tumor_model.keras only if it does not exist.
"""
import os
import ssl
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILENAME = "brain_tumor_model.keras"
# Direct HTTPS URL (GitHub raw) for a brain tumor classification model
MODEL_URL = "https://github.com/josephhu1/CNN-Brain-Tumor-MRI-Classification/raw/main/biohack-model.keras"
MIN_MODEL_SIZE = 500_000


def download_model():
    """
    Create models/ if it does not exist.
    Download brain_tumor_model.keras only if it does not already exist.
    Does not re-download if the file is already present.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)

    if os.path.exists(model_path) and os.path.getsize(model_path) >= MIN_MODEL_SIZE:
        return

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as resp:
            data = resp.read()
        if len(data) < MIN_MODEL_SIZE or data.startswith(b"<!") or data.startswith(b"import "):
            return
        with open(model_path, "wb") as f:
            f.write(data)
    except Exception:
        pass


if __name__ == "__main__":
    download_model()
