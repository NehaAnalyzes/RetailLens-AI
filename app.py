import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RetailLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0a1628 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #f0abfc, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
}
.hero-sub { font-size: 1rem; color: #94a3b8; font-weight: 300; margin: 0; }
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.result-card {
    background: rgba(15,15,30,0.9);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.result-pass { border-color: rgba(52,211,153,0.4) !important; background: rgba(52,211,153,0.05) !important; }
.result-fail { border-color: rgba(248,113,113,0.4) !important; background: rgba(248,113,113,0.05) !important; }
.verdict-pass { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #34d399; }
.verdict-fail { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #f87171; }
.score-num { font-family: 'Syne', sans-serif; font-size: 3.2rem; font-weight: 800; line-height: 1; }
.score-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; }
.check-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
}
.check-name { font-size: 0.9rem; color: #94a3b8; }
.check-score { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; }
.category-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(236,72,153,0.2));
    border: 1px solid rgba(99,102,241,0.3);
    color: #c4b5fd;
    padding: 8px 20px;
    border-radius: 100px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.04em;
}
.conf-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 6px;
    margin: 6px 0;
    overflow: hidden;
}
.stat-box {
    background: rgba(15,15,30,0.8);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.stat-num { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #a5b4fc; }
.stat-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }
div[data-testid="stFileUploader"] {
    background: rgba(15,15,30,0.6);
    border: 2px dashed rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 8px;
}
div[data-testid="stFileUploader"]:hover { border-color: rgba(99,102,241,0.6); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# QUALITY CHECK CONSTANTS
# ─────────────────────────────────────────────
MIN_WIDTH, MIN_HEIGHT = 100, 100
BLUR_THRESHOLD        = 80.0
BRIGHT_MIN, BRIGHT_MAX = 40, 230

CLASS_NAMES = [
    "BABY_PRODUCTS", "BEAUTY_HEALTH", "CLOTHING_ACCESSORIES_JEWELLERY",
    "ELECTRONICS", "GROCERY", "HOBBY_ARTS_STATIONERY",
    "HOME_KITCHEN_TOOLS", "PET_SUPPLIES", "SPORTS_OUTDOOR"
]

CLASS_DISPLAY = {
    "BABY_PRODUCTS":                "👶 Baby Products",
    "BEAUTY_HEALTH":                "💄 Beauty & Health",
    "CLOTHING_ACCESSORIES_JEWELLERY": "👗 Clothing & Accessories",
    "ELECTRONICS":                  "📱 Electronics",
    "GROCERY":                      "🛒 Grocery",
    "HOBBY_ARTS_STATIONERY":        "🎨 Hobby & Arts",
    "HOME_KITCHEN_TOOLS":           "🏠 Home & Kitchen",
    "PET_SUPPLIES":                 "🐾 Pet Supplies",
    "SPORTS_OUTDOOR":               "⚽ Sports & Outdoor",
}

# ─────────────────────────────────────────────
# QUALITY CHECKS
# ─────────────────────────────────────────────
def check_resolution(img):
    h, w = img.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        score = max(0, int(min(w, h) / max(MIN_WIDTH, MIN_HEIGHT) * 100))
        return False, score, f"Too small ({w}×{h}px)"
    return True, min(100, int((min(w, h) / 224) * 100)), f"{w}×{h}px"

def check_blur(img):
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    score    = min(100, int((variance / (BLUR_THRESHOLD * 2)) * 100))
    if variance < BLUR_THRESHOLD:
        return False, score, f"Blurry ({variance:.0f})"
    return True, score, f"Sharp ({variance:.0f})"

def check_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mb   = gray.mean()
    if mb < BRIGHT_MIN:
        return False, max(0, int((mb / BRIGHT_MIN) * 50)), f"Too dark ({mb:.0f})"
    if mb > BRIGHT_MAX:
        over  = mb - BRIGHT_MAX
        score = max(0, int(50 - (over / (255 - BRIGHT_MAX)) * 50))
        return False, score, f"Overexposed ({mb:.0f})"
    mid   = (BRIGHT_MIN + BRIGHT_MAX) / 2
    dist  = abs(mb - mid) / ((BRIGHT_MAX - BRIGHT_MIN) / 2)
    return True, min(100, int((1 - dist * 0.3) * 100)), f"Good ({mb:.0f})"

def run_quality_check(pil_image):
    img  = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    rp, rs, rm = check_resolution(img)
    bp, bs, bm = check_blur(img)
    lp, ls, lm = check_brightness(img)
    overall    = int(rs * 0.30 + bs * 0.40 + ls * 0.30)
    issues     = []
    if not rp: issues.append(rm)
    if not bp: issues.append(bm)
    if not lp: issues.append(lm)
    return {
        "passed":  len(issues) == 0,
        "score":   overall,
        "issues":  issues,
        "checks": {
            "Resolution": {"passed": rp, "score": rs, "msg": rm},
            "Sharpness":  {"passed": bp, "score": bs, "msg": bm},
            "Brightness": {"passed": lp, "score": ls, "msg": lm},
        }
    }

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import gdown, os
        model_path = "efficientnet_b2_ecommerce_best.pth"
        # Download from Google Drive if not present
        if not os.path.exists(model_path):
            file_id = st.secrets.get("MODEL_FILE_ID", "")
            if file_id:
                gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
            else:
                return None, None

        ckpt  = torch.load(model_path, map_location="cpu")
        model = models.efficientnet_b2()
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, 9))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt.get("class_names", CLASS_NAMES)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify(model, class_names, pil_image):
    inp    = transform(pil_image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.softmax(logits, dim=1)[0]
    top5   = probs.topk(5)
    return [
        {"class": class_names[i], "confidence": probs[i].item()}
        for i in top5.indices
    ]

# ─────────────────────────────────────────────
# SCORE COLOR
# ─────────────────────────────────────────────
def score_color(score):
    if score >= 85: return "#34d399"
    if score >= 65: return "#fbbf24"
    return "#f87171"

def conf_color(conf):
    if conf >= 0.7: return "#34d399"
    if conf >= 0.4: return "#fbbf24"
    return "#a5b4fc"

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🔍 AI-Powered · EfficientNet-B2 · 9 Categories</div>
    <div class="hero-title">RetailLens AI</div>
    <p class="hero-sub">Retail Product Image Quality & Category Classification for E-commerce</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STATS ROW
# ─────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="stat-box"><div class="stat-num">9</div><div class="stat-label">Categories</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-box"><div class="stat-num">91/100</div><div class="stat-label">Avg Quality Score</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-box"><div class="stat-num">3</div><div class="stat-label">Quality Checks</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model, class_names = load_model()
if model is None:
    st.warning("⚠️ Model not loaded. Add MODEL_FILE_ID to Streamlit secrets to enable classification.")

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
st.markdown("### 📤 Upload Product Image")
uploaded = st.file_uploader(
    "Drop a product image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    pil_image = Image.open(uploaded).convert("RGB")

    col_img, col_results = st.columns([1, 1.6], gap="large")

    with col_img:
        st.markdown("**Preview**")
        st.image(pil_image, use_column_width=True)
        st.markdown(f"<p style='color:#64748b;font-size:0.8rem;'>📁 {uploaded.name} · {pil_image.size[0]}×{pil_image.size[1]}px</p>", unsafe_allow_html=True)

    with col_results:

        # ── QUALITY CHECK ──────────────────────
        st.markdown("**🔍 Image Quality**")
        qr      = run_quality_check(pil_image)
        verdict = "pass" if qr["passed"] else "fail"
        vclass  = "result-pass" if qr["passed"] else "result-fail"
        vtext   = '<span class="verdict-pass">✅ PASS</span>' if qr["passed"] else '<span class="verdict-fail">❌ FAIL</span>'
        sc      = score_color(qr["score"])

        st.markdown(f"""
        <div class="result-card {vclass}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <div>{vtext}</div>
                <div>
                    <span class="score-num" style="color:{sc}">{qr['score']}</span>
                    <span style="color:#64748b;font-size:1rem;">/100</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        for check_name, info in qr["checks"].items():
            icon  = "✅" if info["passed"] else "❌"
            color = score_color(info["score"])
            bar_w = info["score"]
            st.markdown(f"""
            <div class="check-row">
                <span class="check-name">{icon} {check_name}</span>
                <div style="flex:1;margin:0 16px;">
                    <div class="conf-bar-bg">
                        <div style="width:{bar_w}%;height:100%;background:{color};border-radius:100px;"></div>
                    </div>
                </div>
                <span class="check-score" style="color:{color}">{info['score']}/100</span>
            </div>
            <p style="color:#64748b;font-size:0.78rem;margin:2px 0 8px 24px;">{info['msg']}</p>
            """, unsafe_allow_html=True)

        if qr["issues"]:
            issues_html = "".join([f"<li style='color:#fca5a5;font-size:0.85rem;'>{i}</li>" for i in qr["issues"]])
            st.markdown(f"<ul style='margin-top:12px;padding-left:20px;'>{issues_html}</ul>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── CLASSIFICATION ─────────────────────
        st.markdown("**🏷️ Category Classification**")
        if model is not None:
            results = classify(model, class_names, pil_image)
            top     = results[0]
            top_disp = CLASS_DISPLAY.get(top["class"], top["class"])
            top_conf = top["confidence"]
            top_col  = conf_color(top_conf)

            st.markdown(f"""
            <div class="result-card">
                <p class="score-label" style="margin-bottom:8px;">Predicted Category</p>
                <div class="category-pill">{top_disp}</div>
                <div style="margin-top:16px;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#94a3b8;font-size:0.85rem;">Confidence</span>
                        <span style="font-family:'Syne',sans-serif;font-weight:700;color:{top_col}">{top_conf*100:.1f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div style="width:{top_conf*100:.1f}%;height:100%;background:{top_col};border-radius:100px;"></div>
                    </div>
                </div>
                <div style="margin-top:20px;">
                    <p class="score-label" style="margin-bottom:10px;">Top 5 Predictions</p>
            """, unsafe_allow_html=True)

            for r in results:
                disp = CLASS_DISPLAY.get(r["class"], r["class"])
                conf = r["confidence"]
                col  = conf_color(conf)
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                        <span style="font-size:0.82rem;color:#cbd5e1;">{disp}</span>
                        <span style="font-size:0.82rem;font-weight:600;color:{col}">{conf*100:.1f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div style="width:{conf*100:.1f}%;height:100%;background:{col};border-radius:100px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card">
                <p style="color:#64748b;text-align:center;padding:20px 0;">
                    Model not loaded.<br>
                    <span style="font-size:0.85rem;">Add MODEL_FILE_ID to Streamlit secrets.</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:24px;border-top:1px solid rgba(255,255,255,0.05);">
    <p style="color:#334155;font-size:0.82rem;">
        RetailLens AI · EfficientNet-B2 · 9 Categories<br>
        Built with PyTorch & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)