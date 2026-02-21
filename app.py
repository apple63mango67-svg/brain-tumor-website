"""
Brain Tumor Detection Web Application
Flask backend - accepts brain MRI images only, detects tumor presence.
Uploaded images are not stored on disk; only brain images are accepted for detection.
"""
import os
import io
import base64
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Import model utilities and download helper
from download_model import download_model
from model_utils import load_model, preprocess_image, is_brain_image, predict_tumor, get_tumor_explanation

app = Flask(__name__)
app.config['SECRET_KEY'] = 'brain-tumor-detector-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load model once at startup for faster first prediction
model = None

def get_model():
    global model
    if model is None:
        download_model()
        model = load_model()
    return model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files and 'file' not in request.files:
        return render_template('result.html', error='No image file was uploaded.')

    file = request.files.get('image') or request.files.get('file')
    
    if file.filename == '':
        return render_template('result.html', error='No image was selected.')

    if not allowed_file(file.filename):
        return render_template('result.html', 
            error='Invalid file type. Please upload a brain MRI image (PNG, JPG, JPEG, BMP, or TIFF).')

    try:
        # Read image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)

        # Validate it's a brain image — only brain images accepted for detection
        if not is_brain_image(img_array):
            return render_template('result.html',
                error='Only brain images are accepted for detection.')

        filename = secure_filename(file.filename)
        ext = (filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'png')
        mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'
        image_data = f'data:{mime};base64,' + base64.b64encode(image_bytes).decode('utf-8')

        # Load model and predict (model is preloaded at startup)
        model = get_model()
        try:
            in_shape = model.input_shape
            if in_shape and len(in_shape) >= 3:
                h, w = int(in_shape[1]), int(in_shape[2])
                if h > 0 and w > 0:
                    preprocessed = preprocess_image(img_array, (h, w))
                else:
                    preprocessed = preprocess_image(img_array)
            else:
                preprocessed = preprocess_image(img_array)
        except Exception:
            preprocessed = preprocess_image(img_array)
        has_tumor, confidence = predict_tumor(model, preprocessed)
        explanation = get_tumor_explanation(has_tumor, confidence) if has_tumor else None

        return render_template('result.html',
            has_tumor=has_tumor,
            confidence=confidence,
            filename=filename,
            image_data=image_data,
            explanation=explanation)

    except Exception as e:
        return render_template('result.html', 
            error=f'Error processing image: {str(e)}. Please ensure you uploaded a valid brain MRI image.')


if __name__ == '__main__':
    # Preload model at startup so first analyze request is fast
    print('Loading model...')
    get_model()
    print('Model ready.')
    app.run(debug=True)
