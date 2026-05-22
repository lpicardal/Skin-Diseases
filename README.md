# Skin Disease Screening App
## ResNet-50 + BioBERT — Automated Skin Disease Screening
### CS 20/L Final Project | University of Mindanao

---

## Local Setup

### 1. Install dependencies
pip install -r requirements.txt

### 2. Place model weights
Create a `models/` folder in the same directory as `app.py` and copy in:
- best_baseline.pth       (from Google Colab Cell 17 output)
- best_multimodal.pth     (from Google Colab Cell 19 output)

Your folder structure should look like:
skin_disease_app/
├── app.py
├── requirements.txt
├── README.md
└── models/
├── best_baseline.pth
└── best_multimodal.pth

### 3. Run the app
streamlit run app.py

The app will open at http://localhost:8501

---

## Streamlit Cloud Deployment

1. Push your project to a GitHub repository
2. Upload model weights to the repo (use Git LFS for large files)
   OR store them on Google Drive and add a download step
3. Go to https://share.streamlit.io
4. Connect your GitHub repository
5. Set the main file path to: app.py
6. Deploy

Note: First load will take 2-3 minutes as BioBERT downloads (~440MB).
Subsequent loads use Streamlit's cache.

---

## How to Download Models from Google Colab

Run this in a Colab cell after training completes:

from google.colab import files
files.download('best_baseline.pth')
files.download('best_multimodal.pth')

Then place them in your local models/ directory.

---

## Model Details

| Component       | Specification                              |
|----------------|--------------------------------------------|
| Visual Encoder | ResNet-50 (ImageNet pretrained, frozen)    |
| Text Encoder   | BioBERT dmis-lab/biobert-base-cased-v1.1  |
| Fusion         | Dense(512) → ReLU → Dropout(0.3)          |
| Output         | 22-class Softmax                           |
| Test Accuracy  | 94.83% (Multimodal) / 61.19% (Baseline)   |
| Dataset        | Kaggle Skin Disease Dataset (15,444 images)|