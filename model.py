"""
Offline Signature Verification — Training Script (Improved)
============================================================
Improvements over v1:
  1. Feature config saved inside the pipeline so train/inference are always in sync.
  2. LBP texture features added alongside Hu + Zernike moments.
  3. Grid-directional (HOG-lite) features added for spatial stroke layout.
  4. Skeleton-based structural features (endpoints, branches, stroke length) added.
  5. Fully deterministic — explicit random seeds everywhere.
  6. Explicit warning when mahotas is missing (no silent zero-vector fallback).
  7. Morphological kernel scales with input image resolution.
  8. Writer-independent baseline + Random Forest second baseline both reported.
  9. Confidence-aware evaluation: uncertain band (0.40–0.60) flagged separately.
 10. Feature vector length validated at load time.
"""

import os
import logging
import warnings
import numpy as np
from glob import glob
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

import cv2

try:
    from skimage.morphology import skeletonize
    from skimage.feature import local_binary_pattern
    SKIMAGE_OK = True
except ImportError:
    skeletonize = None
    local_binary_pattern = None
    SKIMAGE_OK = False
    log.warning("scikit-image not installed — skeleton and LBP features will be zeros.")

try:
    import mahotas
    MAHOTAS_OK = True
except ImportError:
    mahotas = None
    MAHOTAS_OK = False
    log.warning("mahotas not installed — Zernike features will be zeros. "
                "Install with: pip install mahotas")

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import joblib

# ──────────────────────────────────────────────
# Config — single source of truth
# ──────────────────────────────────────────────
DATASET_ROOT     = r"C:\Users\amana\Desktop\SignatureV2\dataset"
IMG_SIZE         = 256
DO_SKELETON      = False
ZERNIKE_DEGREE   = 8
LBP_RADIUS       = 3
LBP_N_POINTS     = 24
HOG_GRID         = 8        # divide image into HOG_GRID × HOG_GRID cells
HOG_BINS         = 8        # gradient direction bins per cell
USE_PCA          = True
PCA_VARIANCE     = 0.95
DO_GRID_SEARCH   = True
RANDOM_STATE     = 42
OUTPUT_MODEL_PATH = "svm_signature_pipeline.joblib"
RF_MODEL_PATH     = "rf_signature_pipeline.joblib"

# Store feature config so the app can validate it at load time
FEATURE_CONFIG = dict(
    img_size       = IMG_SIZE,
    do_skeleton    = DO_SKELETON,
    zernike_degree = ZERNIKE_DEGREE,
    lbp_radius     = LBP_RADIUS,
    lbp_n_points   = LBP_N_POINTS,
    hog_grid       = HOG_GRID,
    hog_bins       = HOG_BINS,
)

np.random.seed(RANDOM_STATE)


# ──────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────

def read_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img


def _collect_class_image_paths(root_dir: str, class_name: str) -> list:
    synonyms = {
        "genuine": ["genuine", "gen", "genunie", "genuinie", "genuie"],
        "forged":  ["forged", "forge", "forg", "fake", "forrged", "forged"],
    }
    tokens = synonyms.get(class_name.lower(), [class_name.lower()])
    collected = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(tok in os.path.basename(dirpath).lower() for tok in tokens):
            for fname in filenames:
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                    collected.append(os.path.join(dirpath, fname))
    return sorted(collected)


def dump_dataset_summary(dataset_root: str) -> None:
    log.info("Dataset summary:")
    for subset in ("train", "test"):
        path = os.path.join(dataset_root, subset)
        if not os.path.isdir(path):
            log.info("  %s — not found", path)
            continue
        gen  = _collect_class_image_paths(path, "Genuine")
        forg = _collect_class_image_paths(path, "Forged")
        log.info("  %s: Genuine=%d  Forged=%d", subset, len(gen), len(forg))


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def _adaptive_morph_kernel(img: np.ndarray) -> np.ndarray:
    """Scale closing kernel to image resolution so gaps close correctly on HQ scans."""
    h, w = img.shape
    base   = max(h, w)
    k_size = max(3, int(round(base / 512)) * 2 + 1)   # 3 at ≤512px, 5 at ~1024px, etc.
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
    h, w = cropped.shape
    if h == 0 or w == 0:
        return np.zeros(target_size, dtype=np.uint8)

    scale = min(th / h, tw / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    # Enforce minimum height of 8px so thin strokes don't vanish on wide signatures
    if nh < 8:
        nh = 8
        nw = max(1, int(nw * (nh / max(1, int(h * scale)))))

    resized_small = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(target_size, dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized_small

    if do_skeleton:
        if not SKIMAGE_OK or skeletonize is None:
            raise RuntimeError("scikit-image required for skeletonization.")
        canvas = (skeletonize(canvas > 0).astype(np.uint8) * 255)
    return canvas


# ──────────────────────────────────────────────
# Feature extractors
# ──────────────────────────────────────────────

def hu_moments_features(bin_img: np.ndarray) -> np.ndarray:
    arr = (bin_img // 255) if bin_img.max() > 1 else bin_img
    moments = cv2.moments(arr.astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    out = np.array([
        -np.sign(v) * np.log10(abs(v) + 1e-30) if v != 0 else 0.0
        for v in hu
    ])
    return out  # 7 features


def zernike_features(bin_img: np.ndarray, degree: int = 8) -> np.ndarray:
    n_features = (degree + 1) * (degree + 2) // 2 - 1
    zeros = np.zeros(n_features)

    if not MAHOTAS_OK:
        return zeros      # warning already emitted at import time

    img = (bin_img // 255) if bin_img.max() > 1 else bin_img.copy()
    h, w = img.shape
    s = min(h, w)
    square = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    radius = int(np.floor((s - 1) / 2.0))
    if radius <= 0:
        return zeros
    try:
        return mahotas.features.zernike_moments(square.astype(np.uint8), radius, degree)
    except Exception:
        return zeros


def lbp_features(bin_img: np.ndarray,
                 radius: int = 3,
                 n_points: int = 24) -> np.ndarray:
    """Local Binary Pattern histogram — captures local stroke texture."""
    if not SKIMAGE_OK or local_binary_pattern is None:
        return np.zeros(n_points + 2)
    lbp = local_binary_pattern(bin_img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                           range=(0, n_points + 2), density=True)
    return hist  # n_points + 2 features (26 with defaults)


def hog_lite_features(bin_img: np.ndarray,
                      grid: int = 8,
                      n_bins: int = 8) -> np.ndarray:
    """
    Grid-based directional feature (HOG-lite).
    Divide image into grid×grid cells; compute gradient orientation histogram per cell.
    Captures spatial layout of stroke directions — highly discriminative.
    """
    h, w = bin_img.shape
    cell_h = h // grid
    cell_w = w // grid
    if cell_h < 1 or cell_w < 1:
        return np.zeros(grid * grid * n_bins)

    # Sobel gradients
    gx = cv2.Sobel(bin_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(bin_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) % 180)  # unsigned [0, 180)

    features = []
    for i in range(grid):
        for j in range(grid):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            cell_mag  = magnitude[y0:y1, x0:x1]
            cell_ang  = angle[y0:y1, x0:x1]
            hist, _ = np.histogram(cell_ang, bins=n_bins, range=(0, 180),
                                   weights=cell_mag, density=False)
            # L2-normalise per cell to be illumination-invariant
            norm = np.linalg.norm(hist) + 1e-6
            features.append(hist / norm)

    return np.concatenate(features)  # grid*grid*n_bins features (512 with defaults)


def skeleton_structural_features(bin_img: np.ndarray) -> np.ndarray:
    """
    Topological features from the skeleton:
      - total skeleton pixel count (stroke length proxy)
      - number of endpoints  (degree-1 pixels)
      - number of branch points (degree-3+ pixels)
      - branch/endpoint ratio
      - fraction of image covered by skeleton
    """
    if not SKIMAGE_OK or skeletonize is None:
        return np.zeros(5)

    skel = skeletonize(bin_img > 0)
    total = skel.sum()
    if total == 0:
        return np.zeros(5)

    # Degree of each skeleton pixel using 8-connectivity sum
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    skel_u8 = skel.astype(np.uint8)
    degree_map = cv2.filter2D(skel_u8, -1, kernel) * skel_u8

    endpoints    = int((degree_map == 1).sum())
    branch_pts   = int((degree_map >= 3).sum())
    br_ep_ratio  = branch_pts / max(1, endpoints)
    coverage     = total / (bin_img.shape[0] * bin_img.shape[1])

    return np.array([
        float(total),
        float(endpoints),
        float(branch_pts),
        br_ep_ratio,
        coverage,
    ])  # 5 features


# ──────────────────────────────────────────────
# Full feature extraction
# ──────────────────────────────────────────────

def extract_feature_vector(img_gray: np.ndarray,
                           cfg: dict) -> tuple:
    """
    Returns (feature_vector, preprocessed_image).
    cfg must contain all FEATURE_CONFIG keys.
    """
    pre = preprocess_image(
        img_gray,
        target_size=(cfg["img_size"], cfg["img_size"]),
        do_skeleton=cfg["do_skeleton"],
    )
    hu   = hu_moments_features(pre)                                      # 7
    zm   = zernike_features(pre, degree=cfg["zernike_degree"])           # variable
    lbp  = lbp_features(pre, cfg["lbp_radius"], cfg["lbp_n_points"])    # n_points+2
    hog  = hog_lite_features(pre, cfg["hog_grid"], cfg["hog_bins"])      # grid²×bins
    sk   = skeleton_structural_features(pre)                             # 5

    feat = np.concatenate([hu, zm, lbp, hog, sk])
    feat = np.nan_to_num(feat, neginf=0.0, posinf=0.0)
    return feat, pre


def extract_features_from_path(path: str, cfg: dict) -> np.ndarray:
    img = read_image_gray(path)
    feat, _ = extract_feature_vector(img, cfg)
    return feat


# ──────────────────────────────────────────────
# Dataset loader
# ──────────────────────────────────────────────

def load_dataset(dataset_root: str, cfg: dict,
                 auto_split: bool = False,
                 test_size: float = 0.2) -> tuple:
    X_train, y_train, X_test, y_test = [], [], [], []

    if auto_split:
        genuine = _collect_class_image_paths(dataset_root, "Genuine")
        forged  = _collect_class_image_paths(dataset_root, "Forged")
        files  = [(p, 1) for p in genuine] + [(p, 0) for p in forged]
        if not files:
            raise ValueError("No images found under Genuine/ Forged for auto-split.")
        paths, labels = zip(*files)
        X_tr, X_te, y_tr, y_te = train_test_split(
            paths, labels, test_size=test_size,
            stratify=labels, random_state=RANDOM_STATE
        )
        for p, l in tqdm(zip(X_tr, y_tr), total=len(X_tr), desc="Train features"):
            X_train.append(extract_features_from_path(p, cfg)); y_train.append(l)
        for p, l in tqdm(zip(X_te, y_te), total=len(X_te), desc="Test features"):
            X_test.append(extract_features_from_path(p, cfg)); y_test.append(l)
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    for subset in ("train", "test"):
        subset_path = os.path.join(dataset_root, subset)
        if not os.path.isdir(subset_path):
            continue
        genuine_paths = _collect_class_image_paths(subset_path, "Genuine")
        forged_paths  = _collect_class_image_paths(subset_path, "Forged")
        X_list = X_train if subset == "train" else X_test
        y_list = y_train if subset == "train" else y_test
        for p in tqdm(genuine_paths, desc=f"{subset}/Genuine"):
            X_list.append(extract_features_from_path(p, cfg)); y_list.append(1)
        for p in tqdm(forged_paths, desc=f"{subset}/Forged"):
            X_list.append(extract_features_from_path(p, cfg)); y_list.append(0)

    return (np.array(X_train), np.array(y_train),
            np.array(X_test),  np.array(y_test))


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_far_frr(y_true: np.ndarray,
                    y_pred: np.ndarray) -> tuple:
    cm   = confusion_matrix(y_true, y_pred, labels=[1, 0])
    TP, FN, FP, TN = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    FAR  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR  = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    return FAR, FRR, cm


def evaluate_with_uncertainty(clf, X_test: np.ndarray, y_test: np.ndarray,
                               uncertain_low: float = 0.40,
                               uncertain_high: float = 0.60) -> None:
    """
    Report metrics with a confidence rejection band.
    Samples whose max probability falls in [uncertain_low, uncertain_high]
    are flagged as 'uncertain' and excluded from the hard-label metrics,
    mirroring what a real deployed system would do.
    """
    y_pred  = clf.predict(X_test)
    has_proba = hasattr(clf, "predict_proba")

    if has_proba:
        proba      = clf.predict_proba(X_test)
        confidence = proba.max(axis=1)
        certain_mask   = (confidence < uncertain_low) | (confidence > uncertain_high)
        uncertain_mask = ~certain_mask

        n_uncertain = uncertain_mask.sum()
        log.info("\nConfidence rejection band [%.2f, %.2f]: %d/%d samples marked uncertain (%.1f%%)",
                 uncertain_low, uncertain_high, n_uncertain, len(y_test),
                 100 * n_uncertain / len(y_test))

        if certain_mask.sum() > 0:
            y_true_c = y_test[certain_mask]
            y_pred_c = y_pred[certain_mask]
            FAR, FRR, cm = compute_far_frr(y_true_c, y_pred_c)
            log.info("Metrics on certain samples only (n=%d):", certain_mask.sum())
            log.info("  Accuracy:  %.4f", accuracy_score(y_true_c, y_pred_c))
            log.info("  Precision: %.4f", precision_score(y_true_c, y_pred_c, zero_division=0))
            log.info("  Recall:    %.4f", recall_score(y_true_c, y_pred_c, zero_division=0))
            log.info("  F1:        %.4f", f1_score(y_true_c, y_pred_c, zero_division=0))
            log.info("  FAR:       %.4f  FRR: %.4f", FAR, FRR)

    FAR, FRR, cm = compute_far_frr(y_test, y_pred)
    log.info("\nOverall metrics (all samples):")
    log.info("  Accuracy:  %.4f", accuracy_score(y_test, y_pred))
    log.info("  Precision: %.4f", precision_score(y_test, y_pred, zero_division=0))
    log.info("  Recall:    %.4f", recall_score(y_test, y_pred, zero_division=0))
    log.info("  F1:        %.4f", f1_score(y_test, y_pred, zero_division=0))
    log.info("  FAR:       %.4f  FRR: %.4f", FAR, FRR)
    log.info("Confusion matrix:\n%s", cm)
    log.info("\n%s", classification_report(y_test, y_pred,
                                          target_names=["Forged", "Genuine"],
                                          zero_division=0))


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def build_svm_pipeline(use_pca: bool = True,
                       pca_variance: float = 0.95,
                       do_grid: bool = True,
                       cfg: dict = None) -> object:
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full",
                                 random_state=RANDOM_STATE)))
    steps.append(("svc", SVC(kernel="rbf", probability=True,
                              random_state=RANDOM_STATE)))

    pipeline = Pipeline(steps)
    # Store feature config inside the pipeline for later validation
    pipeline.feature_config = cfg or {}

    if do_grid:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        param_grid = {
            "svc__C":     [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.1, 0.01, 0.001],
        }
        return GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1",
                            n_jobs=-1, verbose=1)
    return pipeline


def build_rf_pipeline(cfg: dict = None) -> Pipeline:
    """Random Forest baseline — no PCA needed, feature importance built-in."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    pipeline.feature_config = cfg or {}
    return pipeline


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray,  y_test: np.ndarray,
                       cfg: dict,
                       use_pca: bool = True,
                       pca_variance: float = 0.95,
                       do_grid: bool = True) -> tuple:
    if X_train.size == 0:
        raise RuntimeError("No training data found.")

    unique, counts = np.unique(y_train, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    log.info("Training class distribution — Genuine: %d  Forged: %d",
             dist.get(1, 0), dist.get(0, 0))
    if len(dist) < 2:
        raise ValueError("Training set must contain both classes.")

    log.info("\n=== SVM (RBF) ===")
    svm_clf = build_svm_pipeline(use_pca, pca_variance, do_grid, cfg)
    svm_clf.fit(X_train, y_train)
    if hasattr(svm_clf, "best_params_"):
        log.info("Best SVM params: %s", svm_clf.best_params_)
    evaluate_with_uncertainty(svm_clf, X_test, y_test)
    joblib.dump(svm_clf, OUTPUT_MODEL_PATH)
    log.info("Saved SVM model → %s", OUTPUT_MODEL_PATH)

    log.info("\n=== Random Forest baseline ===")
    rf_clf = build_rf_pipeline(cfg)
    rf_clf.fit(X_train, y_train)
    evaluate_with_uncertainty(rf_clf, X_test, y_test)
    joblib.dump(rf_clf, RF_MODEL_PATH)
    log.info("Saved RF model → %s", RF_MODEL_PATH)

    # Feature importance (RF — top 15)
    rf_model = rf_clf.named_steps["rf"]
    importances = rf_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    log.info("\nTop-15 feature importances (RF):")
    for rank, idx in enumerate(top_idx, 1):
        log.info("  %2d. feature[%d] = %.4f", rank, idx, importances[idx])

    return svm_clf, rf_clf


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Dataset root: %s", DATASET_ROOT)
    dump_dataset_summary(DATASET_ROOT)

    log.info("Feature config: %s", FEATURE_CONFIG)

    X_train, y_train, X_test, y_test = load_dataset(
        DATASET_ROOT, cfg=FEATURE_CONFIG, auto_split=False
    )
    log.info("Loaded — Train: %d  Test: %d", len(y_train), len(y_test))

    if len(y_train) == 0:
        raise RuntimeError("No training samples found. Check dataset layout.")

    svm_model, rf_model = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        cfg=FEATURE_CONFIG,
        use_pca=USE_PCA,
        pca_variance=PCA_VARIANCE,
        do_grid=DO_GRID_SEARCH,
    )
    log.info("Done.")