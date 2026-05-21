# ============================================================
# app.py — Streamlit Deployment
# ResNet-50 + BioBERT Automated Skin Disease Screening
# CS 20/L Final Project — University of Mindanao
# Aligned with project documentation (Sections 4.3, 5.3, 6)
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel  # type: ignore
import json
import os
from pathlib import Path
import matplotlib  # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # type: ignore

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="Skin Disease Screening | ResNet-50 + BioBERT",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================

BIOBERT_MODEL  = 'dmis-lab/biobert-base-cased-v1.1'
IMG_MEAN       = [0.485, 0.456, 0.406]
IMG_STD        = [0.229, 0.224, 0.225]
NUM_CLASSES    = 22
VISUAL_DIM     = 2048
TEXT_DIM       = 768
FUSION_HIDDEN  = 512
DROPOUT        = 0.3
MAX_TEXT_LEN   = 512

CLASS_NAMES = [
    'Acne', 'Actinic Keratosis', 'Benign Tumors', 'Bullous',
    'Candidiasis', 'Drug Eruption', 'Eczema', 'Infestations/Bites',
    'Lichen', 'Lupus', 'Moles', 'Psoriasis', 'Rosacea',
    'Seborrheic Keratoses', 'Skin Cancer', 'Sun/Sunlight Damage',
    'Tinea', 'Unknown/Normal', 'Vascular Tumors',
    'Vasculitis', 'Vitiligo', 'Warts'
]

# Risk level for each class (used for urgency badge in UI)
CLASS_RISK = {
    'Acne'               : 'low',
    'Actinic Keratosis'  : 'high',
    'Benign Tumors'      : 'low',
    'Bullous'            : 'high',
    'Candidiasis'        : 'medium',
    'Drug Eruption'      : 'high',
    'Eczema'             : 'medium',
    'Infestations/Bites' : 'low',
    'Lichen'             : 'medium',
    'Lupus'              : 'high',
    'Moles'              : 'medium',
    'Psoriasis'          : 'medium',
    'Rosacea'            : 'low',
    'Seborrheic Keratoses': 'low',
    'Skin Cancer'        : 'high',
    'Sun/Sunlight Damage': 'medium',
    'Tinea'              : 'low',
    'Unknown/Normal'     : 'low',
    'Vascular Tumors'    : 'medium',
    'Vasculitis'         : 'high',
    'Vitiligo'           : 'medium',
    'Warts'              : 'low',
}

# Clinical descriptions aligned with BioBERT text corpus (Section 5.2.2.1)
CLINICAL_DESCRIPTIONS = {
    'Acne': (
        "Acne vulgaris is a chronic inflammatory skin condition characterized by comedones, "
        "papules, pustules, nodules, and cysts primarily on the face, back, and chest. "
        "It results from pilosebaceous unit obstruction due to excess sebum, follicular "
        "hyperkeratinization, and Cutibacterium acnes proliferation. Management includes "
        "topical retinoids, benzoyl peroxide, antibiotics, and isotretinoin for severe cases. "
        "Early treatment prevents scarring and psychosocial distress."
    ),
    'Actinic Keratosis': (
        "Actinic keratosis is a premalignant epidermal lesion caused by prolonged UV radiation "
        "exposure, appearing as rough, scaly patches on sun-exposed areas including face, scalp, "
        "ears, and hands. Characterized by atypical keratinocyte proliferation with dysplasia. "
        "Carries risk of progression to squamous cell carcinoma. Treatment includes cryotherapy, "
        "topical fluorouracil, imiquimod, and photodynamic therapy. Regular dermatological "
        "monitoring is essential."
    ),
    'Benign Tumors': (
        "Benign skin tumors include dermatofibromas, lipomas, epidermal inclusion cysts, and "
        "hemangiomas. Characterized by well-defined borders, slow growth, and absence of invasion "
        "or metastasis. Diagnosis is primarily clinical with dermoscopy and biopsy for uncertain "
        "cases. Treatment not required unless symptomatic; surgical excision or laser therapy "
        "are options when needed."
    ),
    'Bullous': (
        "Bullous disorders are autoimmune blistering conditions including bullous pemphigoid and "
        "pemphigus vulgaris, characterized by large fluid-filled blisters due to autoantibody-mediated "
        "destruction of epidermal adhesion proteins. Diagnosis requires immunofluorescence and "
        "serology. Management involves systemic corticosteroids, immunosuppressants, and rituximab "
        "for refractory cases."
    ),
    'Candidiasis': (
        "Cutaneous candidiasis is a fungal infection by Candida species affecting intertriginous "
        "areas including groin, axillae, and inframammary folds. Presents as erythematous macerated "
        "plaques with satellite pustules. Risk factors include immunosuppression, diabetes, and "
        "antibiotic use. Treatment includes topical azole antifungals; systemic therapy for "
        "refractory or disseminated cases."
    ),
    'Drug Eruption': (
        "Drug-induced skin eruptions are adverse cutaneous reactions from medications, ranging "
        "from morbilliform rashes to severe Stevens-Johnson syndrome and toxic epidermal necrolysis. "
        "Most commonly presents as symmetrical erythematous macules and papules. Management requires "
        "discontinuation of the offending drug, antihistamines for mild reactions, and systemic "
        "corticosteroids for severe cases."
    ),
    'Eczema': (
        "Atopic dermatitis is a chronic relapsing inflammatory condition with intense pruritus, "
        "erythema, vesicles, weeping, and lichenification affecting flexural areas. Associated "
        "with IgE sensitization, skin barrier dysfunction from filaggrin mutations, and Th2-mediated "
        "immune dysregulation. Management includes emollients, topical corticosteroids, calcineurin "
        "inhibitors, and dupilumab for severe cases."
    ),
    'Infestations/Bites': (
        "Skin infestations include scabies caused by mites presenting with intense nocturnal "
        "pruritus and linear burrows, and lice infestations. Insect bite reactions manifest as "
        "erythematous papules and urticaria. Scabies treatment includes topical permethrin or "
        "oral ivermectin. Antihistamines and topical steroids are used for symptomatic bite "
        "reactions. All household contacts should be treated simultaneously."
    ),
    'Lichen': (
        "Lichen planus is a chronic inflammatory condition affecting skin, mucous membranes, "
        "and nails. Cutaneous lesions are pruritic, planar, polygonal, purple papules with "
        "Wickham striae visible on dermoscopy. Pathogenesis involves cytotoxic T-cell attack "
        "on basal keratinocytes. Management includes topical corticosteroids, calcineurin "
        "inhibitors, and hydroxychloroquine for widespread disease."
    ),
    'Lupus': (
        "Cutaneous lupus erythematosus presents with photosensitive erythematous scaling plaques "
        "and a butterfly malar rash across the cheeks. Associated with systemic lupus erythematosus. "
        "Histology shows interface dermatitis and vacuolar basal layer degeneration. Management "
        "involves sun protection, hydroxychloroquine antimalarials, topical corticosteroids, "
        "and immunosuppressants for systemic involvement."
    ),
    'Moles': (
        "Melanocytic nevi are benign melanocyte proliferations presenting as uniformly pigmented "
        "well-circumscribed macules, papules, or nodules. ABCDE criteria (Asymmetry, Border, Color, "
        "Diameter, Evolution) help differentiate benign nevi from melanoma. Regular dermoscopic "
        "monitoring recommended. Excision indicated for atypical or dysplastic nevi."
    ),
    'Psoriasis': (
        "Psoriasis is a chronic immune-mediated disease presenting as well-demarcated erythematous "
        "plaques covered by silvery-white scales affecting scalp, elbows, and knees. Pathogenesis "
        "involves Th17-mediated keratinocyte hyperproliferation. Associated with psoriatic arthritis "
        "and cardiovascular disease. Treatment includes topical corticosteroids, phototherapy, "
        "methotrexate, and biologics targeting TNF-alpha, IL-17, and IL-23."
    ),
    'Rosacea': (
        "Rosacea is a chronic facial inflammatory condition with persistent central facial erythema, "
        "telangiectasia, and inflammatory papules. Divided into erythematotelangiectatic, "
        "papulopustular, phymatous, and ocular subtypes. Triggers include UV exposure, hot "
        "beverages, and alcohol. Treatment includes topical metronidazole, azelaic acid, "
        "oral doxycycline, and laser therapy for telangiectasia."
    ),
    'Seborrheic Keratoses': (
        "Seborrheic keratoses are benign epidermal lesions common in elderly presenting as waxy "
        "stuck-on appearing plaques with verrucous surface and variable brown pigmentation. "
        "Show acanthosis, papillomatosis, and horn cysts histologically. The Leser-Trelat sign "
        "may indicate internal malignancy. Treatment includes cryotherapy, curettage, or laser "
        "ablation when symptomatic."
    ),
    'Skin Cancer': (
        "Skin cancers include basal cell carcinoma presenting as pearly nodules, squamous cell "
        "carcinoma as hyperkeratotic plaques, and melanoma as asymmetric variably pigmented lesions. "
        "Melanoma carries highest metastatic potential. Diagnosis requires dermoscopy and biopsy. "
        "Treatment includes surgical excision, Mohs surgery, immunotherapy with pembrolizumab, "
        "and targeted therapy for advanced disease."
    ),
    'Sun/Sunlight Damage': (
        "Solar damage includes photoaging changes such as solar lentigines, telangiectasia, "
        "elastosis, and poikiloderma from repeated UV exposure. UV-B and UV-A cause DNA mutations "
        "and collagen degradation. Prevention requires broad-spectrum SPF 50+ sunscreen. Treatments "
        "include topical retinoids, chemical peels, laser resurfacing, and antioxidants."
    ),
    'Tinea': (
        "Tinea dermatophytosis is a superficial fungal infection by Trichophyton, Microsporum, "
        "and Epidermophyton species. Presents as annular scaly erythematous plaques with peripheral "
        "scaling and central clearing. Classified as capitis, corporis, pedis, unguium, and cruris. "
        "Diagnosis by KOH preparation. Treatment: topical azoles or terbinafine; oral antifungals "
        "for extensive or nail disease."
    ),
    'Unknown/Normal': (
        "Normal healthy skin presents with intact barrier function, even pigmentation, and smooth "
        "texture without inflammatory or neoplastic pathology. Unknown lesions require further "
        "clinical investigation, dermoscopy, and potentially biopsy. Regular self-examination "
        "using ABCDE criteria recommended. Consultation with a board-certified dermatologist "
        "advised for any lesion of uncertain etiology."
    ),
    'Vascular Tumors': (
        "Vascular tumors include infantile hemangiomas, pyogenic granulomas, and angiokeratomas. "
        "Infantile hemangiomas are bright red compressible lesions that involute over time. "
        "Pyogenic granulomas are rapidly growing friable red papules prone to bleeding. Treatment "
        "includes propranolol for hemangiomas, pulsed-dye laser therapy, surgical excision, "
        "and sclerotherapy for vascular malformations."
    ),
    'Vasculitis': (
        "Cutaneous vasculitis presents as palpable purpura, petechiae, livedo reticularis, nodules, "
        "or ulcers on dependent areas due to blood vessel wall inflammation. Classified by vessel "
        "size including IgA vasculitis and ANCA-associated vasculitis. Diagnosis requires biopsy "
        "and direct immunofluorescence. Management includes colchicine, dapsone, corticosteroids, "
        "and immunosuppressants."
    ),
    'Vitiligo': (
        "Vitiligo is an acquired pigmentary disorder with progressive depigmentation due to "
        "autoimmune melanocyte destruction. Presents as well-demarcated milk-white macules showing "
        "bright fluorescence under Wood's lamp. Associated with thyroid disease and type 1 diabetes. "
        "Management includes topical corticosteroids, narrowband UVB phototherapy, excimer laser, "
        "and JAK inhibitors for active widespread disease."
    ),
    'Warts': (
        "Cutaneous warts are benign HPV-induced epidermal proliferations presenting as "
        "hyperkeratotic papules with punctate black thrombosed capillaries. HPV types 1, 2, 4, "
        "and 27 cause common and plantar warts. Treatment includes salicylic acid, cryotherapy, "
        "electrocautery, laser ablation, and intralesional immunotherapy with candida antigen "
        "for recalcitrant cases."
    ),
}

# Specialist urgency advice per class
URGENCY_ADVICE = {
    'high'  : "⚠️ This condition requires prompt consultation with a board-certified dermatologist.",
    'medium': "ℹ️ Schedule a dermatologist appointment for proper evaluation and treatment.",
    'low'   : "✅ This condition is generally manageable. A dermatologist visit is still recommended."
}

# ============================================================
# MODEL CLASS DEFINITIONS
# (Aligned with Section 5.3 of documentation)
# ============================================================

class BaselineMLP(nn.Module):
    """ResNet-50 visual-only classifier (2048-d → 22 classes)."""
    def __init__(self, visual_dim=2048, hidden=512, num_classes=22, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, vis):
        return self.classifier(vis)


class MultimodalFusionMLP(nn.Module):
    """
    ResNet-50 (2048-d) + BioBERT (768-d) Late-Fusion Classifier.
    Concat → 2816-d → Dense(512) → ReLU → Dropout → 22-class output.
    (Section 5.3 of documentation)
    """
    def __init__(self, visual_dim=2048, text_dim=768, hidden=512,
                 num_classes=22, dropout=0.3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, vis, txt):
        x = torch.cat([vis, txt], dim=1)
        x = self.fusion(x)
        return self.classifier(x)


# ============================================================
# CACHED RESOURCE LOADERS
# (Using st.cache_resource so models load only once)
# ============================================================

@st.cache_resource(show_spinner="Loading ResNet-50 feature extractor...")
def load_resnet_extractor():
    """Load frozen ResNet-50 backbone for visual feature extraction."""
    extractor = models.resnet50(pretrained=False)
    extractor.fc = nn.Identity()
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor


@st.cache_resource(show_spinner="Loading BioBERT encoder...")
def load_biobert():
    """Load BioBERT tokenizer and encoder."""
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    model     = AutoModel.from_pretrained(BIOBERT_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


@st.cache_resource(show_spinner="Loading classification models...")
def load_classification_models():
    """Load trained baseline and multimodal MLP checkpoints."""
    baseline_path   = Path("models/best_baseline.pth")
    multimodal_path = Path("models/best_multimodal.pth")

    missing = []
    if not baseline_path.exists():
        missing.append("models/best_baseline.pth")
    if not multimodal_path.exists():
        missing.append("models/best_multimodal.pth")

    if missing:
        return None, None, missing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = BaselineMLP(VISUAL_DIM, FUSION_HIDDEN, NUM_CLASSES, DROPOUT)
    baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    baseline.eval()

    multimodal = MultimodalFusionMLP(
        VISUAL_DIM, TEXT_DIM, FUSION_HIDDEN, NUM_CLASSES, DROPOUT
    )
    multimodal.load_state_dict(torch.load(multimodal_path, map_location=device))
    multimodal.eval()

    return baseline, multimodal, []


@st.cache_resource(show_spinner="Pre-computing BioBERT class embeddings...")
def precompute_text_features(_tokenizer, _biobert_model):
    """
    Pre-extract BioBERT [CLS] embeddings for all 22 class descriptions.
    Cached so this only runs once per session.
    (Section 5.2.2 of documentation)
    """
    text_features = {}
    for class_name in CLASS_NAMES:
        description = CLINICAL_DESCRIPTIONS.get(class_name, "Skin condition.")
        enc = _tokenizer(
            description,
            max_length=MAX_TEXT_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            out = _biobert_model(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask']
            )
            cls_emb = out.last_hidden_state[:, 0, :].squeeze(0)
        text_features[class_name] = cls_emb

    # Stack into (22, 768) matrix
    feat_matrix = torch.stack(
        [text_features[c] for c in CLASS_NAMES], dim=0
    )
    return feat_matrix


# ============================================================
# IMAGE PREPROCESSING
# (Section 5.2.1 of documentation — ImageNet normalization)
# ============================================================

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


# ============================================================
# INFERENCE PIPELINE
# (Sections 4.3 & 5.3 — Sequential dual-model pipeline)
# ============================================================

def run_inference(pil_image, resnet_extractor, baseline_model,
                  multimodal_model, biobert_tokenizer, biobert_model,
                  text_feat_matrix):
    """
    Full two-stage sequential pipeline:
      Stage 1: Image → ResNet-50 extractor → visual features
                     → BaselineMLP → Stage 1 prediction
      Stage 2: BioBERT[predicted class] + visual features
                     → MultimodalFusionMLP → final prediction + guidance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Preprocess image ──
    img_tensor = inference_transform(pil_image).unsqueeze(0).to(device)

    resnet_extractor  = resnet_extractor.to(device)
    baseline_model    = baseline_model.to(device)
    multimodal_model  = multimodal_model.to(device)
    text_feat_matrix  = text_feat_matrix.to(device)

    with torch.no_grad():
        # ── Stage 1A: Visual feature extraction (ResNet-50 GAP → 2048-d) ──
        vis_feat = resnet_extractor(img_tensor)              # (1, 2048)

        # ── Stage 1B: Baseline visual classification ──
        stage1_logits = baseline_model(vis_feat)             # (1, 22)
        stage1_probs  = torch.softmax(stage1_logits, dim=1)  # (1, 22)
        stage1_conf, stage1_idx = stage1_probs.max(dim=1)
        stage1_pred   = CLASS_NAMES[stage1_idx.item()]

        # ── Stage 2: BioBERT text feature for predicted class ──
        txt_feat = text_feat_matrix[stage1_idx].unsqueeze(0) # (1, 768)

        # ── Stage 2: Fusion classification ──
        fusion_logits = multimodal_model(vis_feat, txt_feat)
        fusion_probs  = torch.softmax(fusion_logits, dim=1)
        fusion_conf, fusion_idx = fusion_probs.max(dim=1)
        final_pred    = CLASS_NAMES[fusion_idx.item()]
        final_conf    = fusion_conf.item()

        # ── Top-5 predictions (multimodal) ──
        top5_probs, top5_idx = fusion_probs.topk(5, dim=1)
        top5 = [
            (CLASS_NAMES[i.item()], p.item())
            for i, p in zip(top5_idx[0], top5_probs[0])
        ]

    # ── Clinical guidance from BioBERT corpus ──
    guidance = CLINICAL_DESCRIPTIONS.get(
        final_pred, "Consult a board-certified dermatologist for evaluation."
    )
    risk      = CLASS_RISK.get(final_pred, 'medium')
    urgency   = URGENCY_ADVICE.get(risk, URGENCY_ADVICE['medium'])

    return {
        'final_prediction' : final_pred,
        'final_confidence' : final_conf,
        'stage1_prediction': stage1_pred,
        'stage1_confidence': stage1_conf.item(),
        'top5'             : top5,
        'guidance'         : guidance,
        'risk'             : risk,
        'urgency'          : urgency,
    }


# ============================================================
# HELPER: Confidence bar chart
# ============================================================

def plot_top5(top5_list):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    names  = [c for c, _ in reversed(top5_list)]
    scores = [s * 100 for _, s in reversed(top5_list)]
    bar_colors = ['#1976D2' if i < 4 else '#F57C00' for i in range(len(names))]
    bar_colors = list(reversed(bar_colors))

    bars = ax.barh(names, scores, color=bar_colors, edgecolor='none',
                   height=0.55, alpha=0.9)

    for bar, score in zip(bars, scores):
        ax.text(
            min(score + 1, 105), bar.get_y() + bar.get_height() / 2,
            f'{score:.1f}%', va='center', fontsize=9,
            color='white', fontweight='bold'
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel('Confidence (%)', color='white', fontsize=9)
    ax.set_title('Top-5 Predicted Conditions', color='white',
                 fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    plt.tight_layout()
    return fig


# ============================================================
# CUSTOM CSS
# ============================================================

def inject_css():
    st.markdown("""
    <style>
    /* ── Main background ── */
    .stApp { background-color: #0e1117; }

    /* ── Header banner ── */
    .app-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        border-radius: 12px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(13, 71, 161, 0.4);
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .app-header p {
        color: #bbdefb;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Result cards ── */
    .result-card {
        background: #1a1d2e;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .condition-name {
        font-size: 1.8rem;
        font-weight: 700;
        color: #64b5f6;
        margin: 0;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #a5d6a7;
        margin: 0.2rem 0 0 0;
    }

    /* ── Risk badges ── */
    .badge-high   { background:#b71c1c; color:white; padding:3px 10px;
                    border-radius:20px; font-size:0.78rem; font-weight:700; }
    .badge-medium { background:#e65100; color:white; padding:3px 10px;
                    border-radius:20px; font-size:0.78rem; font-weight:700; }
    .badge-low    { background:#1b5e20; color:white; padding:3px 10px;
                    border-radius:20px; font-size:0.78rem; font-weight:700; }

    /* ── Guidance box ── */
    .guidance-box {
        background: #1e2433;
        border-left: 4px solid #1976D2;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        color: #e0e0e0;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    /* ── Disclaimer ── */
    .disclaimer-box {
        background: #1a1a2e;
        border: 1px solid #ff5722;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #ffccbc;
        font-size: 0.85rem;
        line-height: 1.6;
        margin-top: 1rem;
    }

    /* ── Upload area ── */
    .upload-hint {
        text-align: center;
        color: #546e7a;
        font-size: 0.9rem;
        padding: 1rem;
    }

    /* ── Stage pipeline ── */
    .pipeline-step {
        background: #1a1d2e;
        border: 1px solid #2d3561;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #cfd8dc;
    }
    .pipeline-step .step-label {
        color: #90caf9;
        font-weight: 700;
        margin-right: 0.5rem;
    }

    /* ── Sidebar ── */
    .css-1d391kg { background: #131722; }

    /* ── Metrics ── */
    div[data-testid="metric-container"] {
        background: #1a1d2e;
        border: 1px solid #2d3561;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🩺 About This App")
        st.markdown("""
        This application implements the **ResNet-50 + BioBERT Dual-Model Pipeline**
        for automated skin disease screening, developed as part of the
        **CS 20/L Deep Learning: Computer Vision** final project at the
        **University of Mindanao**.
        """)

        st.divider()
        st.markdown("### 🧠 Model Architecture")
        st.markdown("""
        **Stage 1 — Visual Classification**
        - Model: ResNet-50 CNN
        - Input: 224×224 RGB image
        - Output: 2,048-d visual features → 22-class prediction

        **Stage 2 — Clinical Guidance**
        - Model: BioBERT Transformer
        - Pretrained on PubMed biomedical text
        - Generates evidence-based clinical guidance

        **Fusion**
        - ResNet-50 (2,048-d) + BioBERT (768-d) → 2,816-d
        - Dense(512) → ReLU → Dropout(0.3) → 22-class output
        """)

        st.divider()
        st.markdown("### 📊 Model Performance")
        st.markdown("""
        | Model | Accuracy | F1-Score | AUC-ROC |
        |---|---|---|---|
        | ResNet-50 Only | 61.19% | 0.6119 | 0.9443 |
        | **ResNet+BioBERT** | **94.83%** | **0.9489** | **0.9991** |
        """)

        st.divider()
        st.markdown("### 🗂️ Detectable Conditions")
        for cls in CLASS_NAMES:
            risk = CLASS_RISK.get(cls, 'low')
            icon = "🔴" if risk == "high" else ("🟡" if risk == "medium" else "🟢")
            st.markdown(f"{icon} {cls}")

        st.divider()
        st.markdown("### ⚠️ Disclaimer")
        st.error(
            "This tool is for **educational and screening purposes only**. "
            "It is **not a substitute** for professional medical diagnosis. "
            "Always consult a board-certified dermatologist."
        )
        st.markdown(
            "<small>Dataset: Kaggle Skin Disease Dataset (CC0) | "
            "CS 20/L Final Project | University of Mindanao</small>",
            unsafe_allow_html=True
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    inject_css()

    # ── Header ──
    st.markdown("""
    <div class="app-header">
        <h1>🩺 Automated Skin Disease Screening</h1>
        <p>ResNet-50 CNN + BioBERT Transformer | 22 Skin Disease Categories | University of Mindanao — CS 20/L</p>
    </div>
    """, unsafe_allow_html=True)

    render_sidebar()

    # ── Load models ──
    baseline_model, multimodal_model, missing = load_classification_models()

    if missing:
        st.error(
            f"**Model weights not found.** Place the following files in a `models/` "
            f"folder in the same directory as `app.py`:\n\n"
            + "\n".join([f"- `{m}`" for m in missing])
            + "\n\nDownload these from your Google Colab session after training "
              "(`best_baseline.pth` and `best_multimodal.pth`)."
        )
        st.stop()

    resnet_extractor          = load_resnet_extractor()
    biobert_tokenizer, biobert_model = load_biobert()
    text_feat_matrix          = precompute_text_features(biobert_tokenizer, biobert_model)

    # ── Upload section ──
    st.markdown("### 📤 Upload a Skin Image for Analysis")
    st.markdown(
        "Upload a clear photograph of the skin area of concern. "
        "The model will analyze the image and identify the most likely skin condition."
    )

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP. Max size: 200MB."
    )

    if uploaded_file is None:
        col_hint1, col_hint2, col_hint3 = st.columns(3)
        with col_hint1:
            st.info("📸 **Step 1** — Upload a skin image using the button above")
        with col_hint2:
            st.info("🔬 **Step 2** — ResNet-50 + BioBERT analyzes the image")
        with col_hint3:
            st.info("📋 **Step 3** — Receive prediction and clinical guidance")
        return

    # ── Process uploaded image ──
    try:
        pil_image = Image.open(uploaded_file).convert('RGB')
    except Exception as e:
        st.error(f"Could not open the uploaded image: {e}")
        return

    # ── Layout: image | results ──
    col_img, col_res = st.columns([1, 1.4], gap="large")

    with col_img:
        st.markdown("#### Input Image")
        st.image(pil_image, use_column_width=True,
                 caption=f"Uploaded: {uploaded_file.name}")
        st.markdown(
            f"<small style='color:#546e7a;'>Size: "
            f"{pil_image.width} × {pil_image.height} px | "
            f"Processed to: 224 × 224 px</small>",
            unsafe_allow_html=True
        )

    with col_res:
        with st.spinner("🔬 Analyzing image through ResNet-50 + BioBERT pipeline..."):
            result = run_inference(
                pil_image,
                resnet_extractor,
                baseline_model,
                multimodal_model,
                biobert_tokenizer,
                biobert_model,
                text_feat_matrix
            )

        # ── Pipeline stages ──
        st.markdown("#### Pipeline Result")
        st.markdown(f"""
        <div class="pipeline-step">
            <span class="step-label">Stage 1 — ResNet-50:</span>
            {result['stage1_prediction']}
            <span style="color:#a5d6a7; margin-left:8px;">
                ({result['stage1_confidence']*100:.1f}% visual confidence)
            </span>
        </div>
        <div class="pipeline-step">
            <span class="step-label">Stage 2 — BioBERT Fusion:</span>
            {result['final_prediction']}
            <span style="color:#a5d6a7; margin-left:8px;">
                ({result['final_confidence']*100:.1f}% final confidence)
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Main prediction card ──
        risk  = result['risk']
        badge = f'<span class="badge-{risk}">{risk.upper()} RISK</span>'
        st.markdown(f"""
        <div class="result-card">
            <p class="condition-name">{result['final_prediction']}</p>
            <p class="confidence-text">
                Confidence: {result['final_confidence']*100:.2f}%&nbsp;&nbsp;{badge}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence progress bar ──
        conf_val = result['final_confidence']
        bar_color = ("#ef5350" if conf_val < 0.5
                     else "#ffa726" if conf_val < 0.75 else "#66bb6a")
        st.markdown(f"""
        <div style="margin: -0.5rem 0 0.8rem 0;">
            <div style="background:#2d3561; border-radius:6px; height:10px; overflow:hidden;">
                <div style="background:{bar_color}; width:{conf_val*100:.1f}%;
                     height:100%; border-radius:6px; transition:width 0.4s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Full-width: top-5 chart + guidance ──
    st.divider()
    col_chart, col_guide = st.columns([1, 1.3], gap="large")

    with col_chart:
        st.markdown("#### Top-5 Predictions")
        fig = plot_top5(result['top5'])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction",  result['final_prediction'].split('/')[0])
        m2.metric("Confidence",  f"{result['final_confidence']*100:.1f}%")
        m3.metric("Risk Level",  risk.upper())

    with col_guide:
        st.markdown("#### 🔬 Evidence-Based Clinical Guidance")
        st.markdown(
            f'<div class="guidance-box">{result["guidance"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown("#### 📋 Clinical Recommendation")
        if result['risk'] == 'high':
            st.error(result['urgency'])
        elif result['risk'] == 'medium':
            st.warning(result['urgency'])
        else:
            st.success(result['urgency'])

        st.markdown(f"""
        <div class="disclaimer-box">
            <strong>⚠️ Important Disclaimer</strong><br>
            This AI screening tool is intended for <strong>educational and
            preliminary screening purposes only</strong>. It is <strong>not a
            substitute for professional medical advice, diagnosis, or
            treatment</strong>. The model achieves 94.83% accuracy on the test
            dataset; however, individual predictions may be incorrect.
            Always seek the advice of a licensed dermatologist or qualified
            healthcare provider regarding any skin condition. In the Philippines,
            PhilHealth-accredited dermatologists can be located through your
            nearest hospital or regional health office.
        </div>
        """, unsafe_allow_html=True)

    # ── Expandable per-class details ──
    with st.expander("📖 View All Top-5 Prediction Details"):
        for rank, (cls_name, prob) in enumerate(result['top5'], 1):
            risk_c = CLASS_RISK.get(cls_name, 'low')
            badge  = f'<span class="badge-{risk_c}">{risk_c.upper()}</span>'
            st.markdown(
                f"**{rank}. {cls_name}** — {prob*100:.2f}% &nbsp; {badge}",
                unsafe_allow_html=True
            )
            if rank == 1:
                with st.container():
                    st.caption(CLINICAL_DESCRIPTIONS.get(cls_name, "")[:300] + "...")
            st.markdown("---")


if __name__ == "__main__":
    main()