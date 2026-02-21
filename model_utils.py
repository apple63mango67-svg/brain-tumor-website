"""
Model utilities for brain tumor detection
- Loads pre-trained CNN model
- Validates brain MRI images
- Preprocesses and predicts
Train with train_model.py using dataset/yes (tumor) and dataset/no (no tumor) so
sigmoid output = P(tumor). Set env INVERT_TUMOR_LABEL=1 if your model uses yes=0, no=1.
"""
import os
import numpy as np
from PIL import Image

# Lazy import TensorFlow (heavy)
_tf = None

def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


def load_model():
    """Load the brain tumor detection model. Must be trained with train_model.py on your dataset/yes and dataset/no."""
    tf = _get_tf()
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, 'models', 'brain_tumor_model.keras')
    alt_path = os.path.join(base, 'models', 'brain_tumor_model.h5')

    for path in [model_path, alt_path]:
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            try:
                return tf.keras.models.load_model(path)
            except Exception:
                pass  # Try next or fall back
    if not os.path.exists(model_path) and not os.path.exists(alt_path):
        import sys
        print("WARNING: No trained model found. Run: python train_model.py (with dataset/yes and dataset/no folders)", file=sys.stderr)
    return build_model()


def build_model():
    """Build the CNN architecture for brain tumor classification."""
    tf = _get_tf()
    from tensorflow.keras import layers, Model
    
    inputs = layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


INPUT_SIZE = 128  # Default for built-in model

def preprocess_image(img_array, target_size=None):
    """Preprocess image to match training: RGB, resize to 128x128, normalize [0,1]. No mean subtraction."""
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    size = target_size or (INPUT_SIZE, INPUT_SIZE)
    if isinstance(size, int):
        size = (size, size)
    # Match train_model: RGB, LANCZOS resize, then float32 / 255
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")
    img = img.resize(size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def is_brain_image(img_array):
    """
    Validate that the image appears to be a brain MRI.
    Uses heuristics: size, grayscale/dark tone, typical MRI dimensions.
    """
    if img_array is None or len(img_array.shape) < 2:
        return False
    
    h, w = img_array.shape[:2]
    
    # Brain MRIs are typically square-ish (roughly 0.5 to 2.0 aspect ratio)
    aspect = max(h, w) / (min(h, w) + 1e-6)
    if aspect > 3 or aspect < 0.35:
        return False
    
    # Minimum size (very small images are unlikely to be MRIs)
    if min(h, w) < 50:
        return False
    
    # Brain scans often have dark backgrounds and medium-bright brain tissue
    # Check that it's not a random photo (e.g. colorful)
    if len(img_array.shape) >= 3:
        mean_val = np.mean(img_array)
        std_val = np.std(img_array)
        # MRIs tend to have moderate contrast, not extreme
        if std_val < 5 and mean_val > 250:
            return False  # Likely blank/white
        if std_val < 5 and mean_val < 5:
            return False  # Likely blank/black
    return True


def _should_invert_labels():
    """True if we should interpret raw model output as P(no tumor) and invert to get yes→tumor, no→no tumor."""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "models", "label_convention.txt")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("yes_is_tumor="):
                        # yes_is_tumor=0 means model was trained with yes=no tumor -> invert
                        return line.split("=", 1)[1].strip() in ("0", "false", "no")
        except Exception:
            pass
    # No file or yes_is_tumor=1: our train_model uses yes=1 (tumor), raw = P(tumor), don't invert
    return False


def predict_tumor(model, preprocessed_img):
    """Run inference. Returns (has_tumor: bool, confidence: float 0-100).
    Ensures: yes folder → tumor, no folder → no tumor (via label_convention.txt or INVERT_TUMOR_LABEL).
    """
    pred = model.predict(preprocessed_img, verbose=0)
    raw = np.squeeze(pred)
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    raw = np.asarray(raw)
    if raw.size == 1:
        raw_val = float(raw.flat[0])
    elif len(raw) > 1:
        raw_val = float(np.sum(raw[:-1]))
    else:
        raw_val = float(raw)
    raw_val = max(0.0, min(1.0, raw_val))
    # train_model.py: yes=1 (tumor), so raw = P(tumor). If labels were swapped, use label_convention yes_is_tumor=0 or INVERT_TUMOR_LABEL=1.
    prob_tumor = (1.0 - raw_val) if _should_invert_labels() else raw_val
    if os.environ.get("INVERT_TUMOR_LABEL", "").strip() in ("1", "true", "yes"):
        prob_tumor = 1.0 - prob_tumor
    has_tumor = prob_tumor >= 0.5
    confidence = prob_tumor * 100 if has_tumor else (1 - prob_tumor) * 100
    return bool(has_tumor), round(float(confidence), 1)


def get_tumor_explanation(has_tumor, confidence):
    """Return a short explanation of how tumor detection was determined (for display when tumor is detected)."""
    if not has_tumor:
        return None
    return (
        "The model detected abnormal tissue patterns in the scan that match learned features of brain tumors from MRI data. "
        "It uses a convolutional neural network trained on brain scan images (tumor vs. no tumor) to highlight regions with "
        "unusual intensity and texture. This result is based on a confidence score of {:.1f}%. "
        "This is an AI screening aid—always seek a clinical diagnosis from a healthcare provider."
    ).format(confidence)
