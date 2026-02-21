# Brain Tumor Detector

Upload brain MRI images to detect tumor presence. Uses Flask backend and a CNN model. **Only brain MRI images are accepted** - other images will be rejected.

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get the pre-trained model (choose one):
   - **Option A**: Download pre-trained model: `python download_model.py`
   - **Option B**: Train your own: place the dataset in `dataset/` (with `Training/` and `Testing/` subdirs: glioma, meningioma, pituitary, notumor), then run `python train_model.py`

## Run

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Usage

1. Upload a brain MRI image (PNG, JPG, BMP, or TIFF)
2. Click "Analyze Scan"
3. View result: Tumor Detected or No Tumor Detected, with confidence percentage

## Note

This is an AI-assisted screening tool, not a medical diagnosis. Always consult a healthcare professional.
