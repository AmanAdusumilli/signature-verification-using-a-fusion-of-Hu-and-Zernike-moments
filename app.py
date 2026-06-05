"""
Offline Signature Verification — Streamlit App (v3)
====================================================
Works with BOTH old models (Hu + Zernike only, ~575 features)
AND new models (Hu + Zernike + LBP + HOG + Skeleton, ~594 features).

Key fix: auto-detects the model's expected feature count and builds
the matching feature extraction config automatically — no manual config
needed, no mismatch errors.
"""

import os
import io
import cv2
import joblib
import logging
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

log = logging.getLogger(__name__)

try:
    from skimage.morphology import skeletonize
    from skimage.feature import local_binary_pattern
    SKIMAGE_OK = True
except ImportError:
    skeletonize = None
    local_binary_pattern = None
    SKIMAGE_OK = False

try:
    import mahotas
    MAHOTAS_OK = True
except ImportError:
    mahotas = None
    MAHOTAS_OK = False

st.set_page_config(page_title="Signature Verifier", layout="centered")

UNCERTAIN_LOW  = 0.40
UNCERTAIN_HIGH = 0.60

# ──────────────────────────────────────────────
# Config profiles
# Old model: Hu(7) + Zernike(degree=8 -> 44) = 51 raw features
# New model: Hu(7) + Zernike(44) + LBP(26) + HOG(512) + Skeleton(5) = 594
# ──────────────────────────────────────────────

OLD_CFG = dict(
    img_size       = 256,
    do_skeleton    = False,
    zernike_degree = 8,
    use_lbp        = False,
    use_hog        = False,
    use_skel       = False,
    lbp_radius     = 3,
    lbp_n_points   = 24,
    hog_grid       = 8,
    hog_bins       = 8,
)

NEW_CFG = dict(
    img_size       = 256,
    do_skeleton    = False,
    zernike_degree = 8,
    use_lbp        = True,
    use_hog        = True,
    use_skel       = True,
    lbp_radius     = 3,
    lbp_n_points   = 24,
    hog_grid       = 8,
    hog_bins       = 8,
)


def _zernike_n(degree: int) -> int:
    return (degree + 1) * (degree + 2) // 2 - 1


def _expected_n_features(cfg: dict) -> int:
    n  = 7
    n += _zernike_n(cfg["zernike_degree"])
    if cfg.get("use_lbp", True):
        n += cfg["lbp_n_points"] + 2
    if cfg.get("use_hog", True):
        n += cfg["hog_grid"] ** 2 * cfg["hog_bins"]
    if cfg.get("use_skel", True):
        n += 5
    return n


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def _get_scaler_n_features(model) -> int | None:
    candidates = [model]
    if hasattr(model, "best_estimator_"):
        candidates.append(model.best_estimator_)
    for obj in candidates:
        if hasattr(obj, "named_steps"):
            scaler = obj.named_steps.get("scaler")
            if scaler is not None and hasattr(scaler, "n_features_in_"):
                return int(scaler.n_features_in_)
    return None


@st.cache_resource
def load_model(path: str):
    """
    Load model and auto-detect the correct feature config.
    Priority:
      1. feature_config stored inside the model (new models from code_improved.py)
      2. Auto-match from scaler.n_features_in_ against known configs
      3. Fallback to OLD_CFG
    Returns (model, cfg, detect_mode_string)
    """
    if not os.path.exists(path):
        return None, None, None

    model = joblib.load(path)

    # 1. Stored config (new models)
    stored = getattr(model, "feature_config", None)
    if stored is None and hasattr(model, "best_estimator_"):
        stored = getattr(model.best_estimator_, "feature_config", None)
    if stored is not None:
        stored.setdefault("use_lbp",  True)
        stored.setdefault("use_hog",  True)
        stored.setdefault("use_skel", True)
        return model, stored, "stored config"

    # 2. Auto-detect from scaler
    n = _get_scaler_n_features(model)
    if n is not None:
        if n == _expected_n_features(OLD_CFG):
            return model, OLD_CFG, f"auto-detected legacy ({n} features)"
        if n == _expected_n_features(NEW_CFG):
            return model, NEW_CFG, f"auto-detected new ({n} features)"
        # Unknown size — use old cfg but warn
        return model, OLD_CFG, f"unknown size ({n} features) — using legacy config"

    # 3. Fallback
    return model, OLD_CFG, "fallback to legacy config"


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def read_image_bytes_to_gray(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def _adaptive_morph_kernel(img: np.ndarray) -> np.ndarray:
    base   = max(img.shape)
    k_size = max(3, int(round(base / 512)) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))


def preprocess_image(img: np.ndarray,
                     target_size: tuple = (256, 256),
                     denoise: bool = True,
                     do_skeleton: bool = False) -> np.ndarray:
    if denoise:
        img = cv2.medianBlur(img, 3)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) > 127:
        bw = cv2.bitwise_not(bw)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, _adaptive_morph_kernel(bw))
    coords = cv2.findNonZero(bw)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = bw[y:y + h, x:x + w]
    else:
        cropped = bw
    th, tw = target_size
    h, w   = cropped.shape
    if h == 0 or w == 0:
        return np.zeros(target_size, dtype=np.uint8)
    scale = min(th / h, tw / w)
    nh    = max(8, int(h * scale))
    nw    = max(1, int(w * scale))
    small  = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(target_size, dtype=np.uint8)
    canvas[(th - nh) // 2:(th - nh) // 2 + nh,
           (tw - nw) // 2:(tw - nw) // 2 + nw] = small
    if do_skeleton:
        if not SKIMAGE_OK:
            st.error("scikit-image required for skeletonization.")
            return canvas
        canvas = (skeletonize(canvas > 0).astype(np.uint8) * 255)
    return canvas


# ──────────────────────────────────────────────
# Feature extractors
# ──────────────────────────────────────────────

def hu_moments_features(bin_img: np.ndarray) -> np.ndarray:
    arr = (bin_img // 255) if bin_img.max() > 1 else bin_img
    hu  = cv2.HuMoments(cv2.moments(arr.astype(np.uint8))).flatten()
    return np.array([
        -np.sign(v) * np.log10(abs(v) + 1e-30) if v != 0 else 0.0
        for v in hu
    ])


def zernike_features(bin_img: np.ndarray, degree: int = 8) -> np.ndarray:
    n = _zernike_n(degree)
    if not MAHOTAS_OK:
        return np.zeros(n)
    img = (bin_img // 255) if bin_img.max() > 1 else bin_img.copy()
    h, w = img.shape
    s    = min(h, w)
    sq   = img[(h - s) // 2:(h - s) // 2 + s,
               (w - s) // 2:(w - s) // 2 + s]
    r    = int(np.floor((s - 1) / 2.0))
    if r <= 0:
        return np.zeros(n)
    try:
        return mahotas.features.zernike_moments(sq.astype(np.uint8), r, degree)
    except Exception:
        return np.zeros(n)


def lbp_features(bin_img: np.ndarray,
                 radius: int = 3,
                 n_points: int = 24) -> np.ndarray:
    if not SKIMAGE_OK or local_binary_pattern is None:
        return np.zeros(n_points + 2)
    lbp  = local_binary_pattern(bin_img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                           range=(0, n_points + 2), density=True)
    return hist


def hog_lite_features(bin_img: np.ndarray,
                      grid: int = 8,
                      n_bins: int = 8) -> np.ndarray:
    h, w = bin_img.shape
    ch, cw = h // grid, w // grid
    if ch < 1 or cw < 1:
        return np.zeros(grid * grid * n_bins)
    gx = cv2.Sobel(bin_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(bin_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag   = np.hypot(gx, gy)
    angle = np.degrees(np.arctan2(gy, gx)) % 180
    out   = []
    for i in range(grid):
        for j in range(grid):
            hist, _ = np.histogram(
                angle[i*ch:(i+1)*ch, j*cw:(j+1)*cw],
                bins=n_bins, range=(0, 180),
                weights=mag[i*ch:(i+1)*ch, j*cw:(j+1)*cw],
            )
            out.append(hist / (np.linalg.norm(hist) + 1e-6))
    return np.concatenate(out)


def skeleton_structural_features(bin_img: np.ndarray) -> np.ndarray:
    if not SKIMAGE_OK or skeletonize is None:
        return np.zeros(5)
    skel  = skeletonize(bin_img > 0)
    total = skel.sum()
    if total == 0:
        return np.zeros(5)
    k       = np.ones((3, 3), np.uint8); k[1, 1] = 0
    deg     = cv2.filter2D(skel.astype(np.uint8), -1, k) * skel.astype(np.uint8)
    ep      = int((deg == 1).sum())
    bp      = int((deg >= 3).sum())
    return np.array([float(total), float(ep), float(bp),
                     bp / max(1, ep),
                     total / (bin_img.shape[0] * bin_img.shape[1])])


def extract_feature_vector(img_gray: np.ndarray, cfg: dict) -> tuple:
    pre   = preprocess_image(img_gray,
                             target_size=(cfg["img_size"], cfg["img_size"]),
                             do_skeleton=cfg["do_skeleton"])
    parts = [hu_moments_features(pre),
             zernike_features(pre, cfg["zernike_degree"])]
    if cfg.get("use_lbp", True):
        parts.append(lbp_features(pre, cfg["lbp_radius"], cfg["lbp_n_points"]))
    if cfg.get("use_hog", True):
        parts.append(hog_lite_features(pre, cfg["hog_grid"], cfg["hog_bins"]))
    if cfg.get("use_skel", True):
        parts.append(skeleton_structural_features(pre))
    feat = np.concatenate(parts)
    return np.nan_to_num(feat, neginf=0.0, posinf=0.0), pre


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────

def gradient_overlay(img: np.ndarray) -> np.ndarray:
    gx  = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)


def confidence_gauge(confidence: float, verdict: str) -> None:
    colour = ("normal"  if verdict.startswith("Genuine") else
              "inverse" if verdict.startswith("Forged")  else "off")
    st.metric("Confidence", f"{confidence * 100:.1f}%", delta_color=colour)
    st.progress(int(confidence * 100))


def feature_importance_chart(model, cfg: dict) -> None:
    pipe = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    if not (hasattr(pipe, "named_steps") and "rf" in pipe.named_steps):
        return
    rf          = pipe.named_steps["rf"]
    importances = rf.feature_importances_
    zm_n        = _zernike_n(cfg["zernike_degree"])
    names       = [f"Hu_{i}" for i in range(7)]
    names      += [f"Zernike_{i}" for i in range(zm_n)]
    if cfg.get("use_lbp"):
        names += [f"LBP_{i}" for i in range(cfg["lbp_n_points"] + 2)]
    if cfg.get("use_hog"):
        names += [f"HOG_{i}" for i in range(cfg["hog_grid"] ** 2 * cfg["hog_bins"])]
    if cfg.get("use_skel"):
        names += ["Skel_len", "Skel_ep", "Skel_bp", "Skel_ratio", "Skel_cov"]
    top   = min(20, len(importances))
    idx   = np.argsort(importances)[::-1][:top]
    lbls  = [names[i] if i < len(names) else f"f{i}" for i in idx]
    cmap  = {"Hu": "#4CAF50", "Zernike": "#2196F3",
              "LBP": "#FF9800", "HOG": "#9C27B0", "Skel": "#F44336"}
    clrs  = [cmap.get(l.split("_")[0], "#888") for l in lbls]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(lbls[::-1], importances[idx][::-1], color=clrs[::-1])
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title(f"Top {top} feature importances (RF)", fontsize=11)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

st.title("Offline Signature Verification")
st.caption("Auto-detects feature config from loaded model — works with old and new .joblib files")

# ── Sidebar ──
st.sidebar.header("Model")
model_path = st.sidebar.text_input("Model path",
                                   value="svm_signature_pipeline.joblib")

if st.sidebar.button("Load / reload model") or "model" not in st.session_state:
    with st.spinner("Loading model…"):
        _m, _cfg, _mode = load_model(model_path)
    if _m is not None:
        st.session_state.update(model=_m, cfg=_cfg,
                                model_path=model_path, detect_mode=_mode)
    else:
        for k in ("model", "cfg", "model_path", "detect_mode"):
            st.session_state.pop(k, None)

model       = st.session_state.get("model")
cfg         = st.session_state.get("cfg")
detect_mode = st.session_state.get("detect_mode", "")

if model is not None and cfg is not None:
    n_features = _expected_n_features(cfg)
    n_model    = _get_scaler_n_features(model)
    feature_mode = ("New  (Hu + Zernike + LBP + HOG + Skeleton)"
                    if cfg.get("use_lbp") else "Legacy  (Hu + Zernike only)")
    st.sidebar.success("Model loaded ✓")
    st.sidebar.info(
        f"**Mode:** {feature_mode}\n\n"
        f"**Detection:** {detect_mode}\n\n"
        f"**Features:** {n_features}\n\n"
        f"**Image size:** {cfg['img_size']}×{cfg['img_size']}\n\n"
        f"**Zernike degree:** {cfg['zernike_degree']}"
    )
    if n_model is not None and n_model != n_features:
        st.sidebar.error(
            f"Mismatch: model expects {n_model} but config gives {n_features}. "
            "Please retrain with code_improved.py."
        )
    if not MAHOTAS_OK:
        st.sidebar.warning("mahotas not installed → Zernike = zeros.\n"
                           "`pip install mahotas`")
    if not SKIMAGE_OK and cfg.get("use_lbp"):
        st.sidebar.warning("scikit-image not installed → LBP/Skeleton = zeros.\n"
                           "`pip install scikit-image`")
else:
    st.sidebar.warning(f"No model found at `{model_path}`.")

st.sidebar.markdown("---")
show_pre  = st.sidebar.checkbox("Show preprocessed image", value=True)
show_grad = st.sidebar.checkbox("Show gradient overlay",   value=True)
show_imp  = st.sidebar.checkbox("Show feature importance (RF only)", value=True)

# ── Predict ──
st.header("Predict")
uploaded = st.file_uploader("Upload a signature image",
                            type=["png", "jpg", "jpeg", "tiff", "bmp"])

if uploaded is not None:
    img_bytes = uploaded.read()
    img_gray  = read_image_bytes_to_gray(img_bytes)

    if img_gray is None:
        st.error("Could not decode image.")
        st.stop()

    if model is None or cfg is None:
        st.error("No model loaded. Use the sidebar to load a model first.")
        st.stop()

    with st.spinner("Extracting features…"):
        try:
            feat, pre = extract_feature_vector(img_gray, cfg)
        except Exception as e:
            st.error(f"Feature extraction failed: {e}")
            st.stop()

    # Images row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Original")
        st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
    if show_pre:
        with c2:
            st.caption("Preprocessed")
            st.image(Image.fromarray(pre), use_container_width=True)
    if show_grad:
        with c3:
            st.caption("Gradient magnitude")
            st.image(Image.fromarray(
                cv2.cvtColor(gradient_overlay(pre), cv2.COLOR_BGR2RGB)
            ), use_container_width=True)

    st.markdown("---")

    # Predict
    try:
        pred       = model.predict([feat])[0]
        confidence = (float(model.predict_proba([feat])[0].max())
                      if hasattr(model, "predict_proba") else None)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if confidence is not None and UNCERTAIN_LOW <= confidence <= UNCERTAIN_HIGH:
        verdict = "Uncertain"
    else:
        verdict = "Genuine ✓" if pred == 1 else "Forged ✗"

    if verdict.startswith("Genuine"):
        st.success(f"### {verdict}")
    elif verdict.startswith("Forged"):
        st.error(f"### {verdict}")
    else:
        st.warning(f"### ⚠️ {verdict} — manual review recommended")

    if confidence is not None:
        confidence_gauge(confidence, verdict)

    # Feature breakdown
    with st.expander("Feature vector breakdown"):
        zm_n  = _zernike_n(cfg["zernike_degree"])
        n_lbp = cfg["lbp_n_points"] + 2
        n_hog = cfg["hog_grid"] ** 2 * cfg["hog_bins"]
        off   = 0
        ca, cb = st.columns(2)
        with ca:
            st.metric("Hu Moments",
                      f"7 features | mean {feat[off:off+7].mean():.4f}")
            off += 7
            st.metric(f"Zernike (deg={cfg['zernike_degree']})",
                      f"{zm_n} features | mean {feat[off:off+zm_n].mean():.4f}")
            off += zm_n
        with cb:
            if cfg.get("use_lbp"):
                st.metric("LBP texture",
                          f"{n_lbp} features | mean {feat[off:off+n_lbp].mean():.4f}")
                off += n_lbp
                st.metric("HOG-lite",
                          f"{n_hog} features | mean {feat[off:off+n_hog].mean():.4f}")
                off += n_hog
                st.metric("Skeleton (5)",
                          str(feat[-5:].round(4).tolist()))
            st.metric("Total features", str(len(feat)))

    if show_imp:
        feature_importance_chart(model, cfg)

    if st.button("Save preprocessed image"):
        out = f"preprocessed_{uploaded.name}"
        Image.fromarray(pre).save(out)
        st.success(f"Saved → {out}")

st.markdown("---")
st.caption("Compatible with legacy models (Hu + Zernike) and "
           "new models (+ LBP + HOG-lite + Skeleton). "
           "Feature config auto-detected from loaded model.")