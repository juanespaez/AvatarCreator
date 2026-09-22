"""
Local, Colab-free mirror of Pipeline.ipynb cells 2-5: scan the data folders,
extract landmarks, crop each region, and save a preview grid per query photo.

This is the very first check the notebook itself recommends before doing
anything expensive (building embeddings/indices) - see Pipeline.ipynb cell 9's
markdown. Run this locally to sanity-check CROP_CFG against real photos
without needing Colab, Drive, ArcFace, or CLIP.

Usage:
    .venv/bin/python scripts/preview_crops.py [photo1.jpg photo2.jpg ...]

With no arguments, it previews every image found in queries/.
Output PNGs are written to _cache/previews/.
"""
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PIECES_DIR = ROOT / "pieces"
REFS_DIR = ROOT / "refs"
QUERIES_DIR = ROOT / "queries"
OUT_DIR = ROOT / "_cache" / "previews"

CATEGORIES = ["face_shape", "skin_tone", "eyes", "eyebrows", "nose", "mouth", "beard", "hair"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def scan_bank():
    bank = {}
    for cat in CATEGORIES:
        d = PIECES_DIR / cat
        items = sorted((p.stem, p) for p in d.glob("*") if p.suffix.lower() in IMG_EXT) if d.exists() else []
        bank[cat] = items
    return bank


def scan_refs():
    refs = {}
    for cat in CATEGORIES:
        d = REFS_DIR / cat
        items = []
        if d.exists():
            for piece_dir in sorted(d.iterdir()):
                if piece_dir.is_dir():
                    items += [
                        (piece_dir.name, p)
                        for p in sorted(piece_dir.glob("*"))
                        if p.suffix.lower() in IMG_EXT
                    ]
        refs[cat] = items
    return refs


def print_data_report():
    bank, refs = scan_bank(), scan_refs()
    print(f"\n{'category':<12}{'pieces':>8}{'refs':>8}{'refs/piece':>12}")
    for cat in CATEGORIES:
        n_p, n_r = len(bank[cat]), len(refs[cat])
        ids = len({i for i, _ in refs[cat]})
        print(f"{cat:<12}{n_p:>8}{n_r:>8}{(n_r / ids if ids else 0):>12.1f}")
    for cat in CATEGORIES:
        piece_ids = {i for i, _ in bank[cat]}
        orphans = {i for i, _ in refs[cat]} - piece_ids
        if orphans:
            print(f"WARNING {cat}: refs reference unknown piece_ids -> {sorted(orphans)[:5]}")
    print()


# ---- landmarks (mirrors Pipeline.ipynb cell 6) ------------------------------
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
_FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_landmarks(rgb):
    res = _FACE_MESH.process(rgb)
    if not res.multi_face_landmarks:
        return None
    h, w = rgb.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)


REGION_IDX = {
    "eyes": [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7,
             362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249],
    "eyebrows": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
                 300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "nose": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 115, 344, 220, 440],
    "mouth": [61, 291, 0, 17, 13, 14, 78, 308, 37, 267, 84, 314, 181, 405, 146, 375],
    "face_shape": list(range(0, 17)) + [234, 454, 10, 152, 127, 356, 132, 361, 172, 397],
    "beard": [61, 291, 17, 18, 200, 199, 175, 152, 148, 176, 149, 150, 136, 172, 58, 132,
              377, 400, 378, 379, 365, 397, 288, 361],
    "hair": [10, 109, 338, 67, 297, 54, 284, 21, 251, 103, 332],
    "skin_tone": [50, 280, 205, 425, 101, 330, 116, 345],
}

# ---- crops (mirrors Pipeline.ipynb cell 8) ----------------------------------
L_EYE_OUT, R_EYE_OUT = 263, 33


def align_face(rgb, lms):
    p_l, p_r = lms[L_EYE_OUT], lms[R_EYE_OUT]
    dy, dx = (p_l - p_r)[1], (p_l - p_r)[0]
    angle = np.degrees(np.arctan2(dy, dx))
    center = tuple(((p_l + p_r) / 2).tolist())
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    h, w = rgb.shape[:2]
    out = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    lms_h = np.hstack([lms, np.ones((len(lms), 1), np.float32)])
    return out, (lms_h @ M.T).astype(np.float32)


def iod(lms):
    return float(np.linalg.norm(lms[L_EYE_OUT] - lms[R_EYE_OUT]))


CROP_CFG = {
    "eyes": dict(pad=0.30, dy=0.00),
    "eyebrows": dict(pad=0.35, dy=0.00),
    "nose": dict(pad=0.25, dy=0.00),
    "mouth": dict(pad=0.30, dy=0.00),
    "face_shape": dict(pad=0.10, dy=0.00),
    "beard": dict(pad=0.20, dy=0.10),
    "hair": dict(pad=0.90, dy=-0.75),
    "skin_tone": dict(pad=0.15, dy=0.00),
}


def crop_region(rgb_aligned, lms_aligned, category, out_size=224) -> Optional[np.ndarray]:
    idx = REGION_IDX[category]
    pts = lms_aligned[idx]
    cfg = CROP_CFG[category]
    s = iod(lms_aligned)

    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + cfg["dy"] * s
    half = max(x1 - x0, y1 - y0) / 2 + cfg["pad"] * s

    h, w = rgb_aligned.shape[:2]
    X0, Y0 = int(round(cx - half)), int(round(cy - half))
    X1, Y1 = int(round(cx + half)), int(round(cy + half))

    pl, pt = max(0, -X0), max(0, -Y0)
    pr, pb = max(0, X1 - w), max(0, Y1 - h)
    if pl or pt or pr or pb:
        rgb_aligned = cv2.copyMakeBorder(rgb_aligned, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        X0 += pl
        X1 += pl
        Y0 += pt
        Y1 += pt

    patch = rgb_aligned[Y0:Y1, X0:X1]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_CUBIC)


def crops_from_photo(path, categories=None, out_size=224):
    rgb = load_rgb(path)
    lms = get_landmarks(rgb)
    if lms is None:
        return {}
    a_rgb, a_lms = align_face(rgb, lms)
    cats = categories or CATEGORIES
    out = {}
    for c in cats:
        p = crop_region(a_rgb, a_lms, c, out_size)
        if p is not None:
            out[c] = p
    return out


def preview_crops(path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crops = crops_from_photo(path)
    if not crops:
        print(f"NO FACE DETECTED: {path}")
        return
    fig, axes = plt.subplots(1, len(crops), figsize=(2.0 * len(crops), 2.4))
    axes = np.atleast_1d(axes)
    for ax, (c, img) in zip(axes, crops.items()):
        ax.imshow(img)
        ax.set_title(c, fontsize=9)
        ax.axis("off")
    fig.suptitle(Path(path).name, fontsize=10)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{Path(path).stem}_crops.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"OK  {path.name} -> {dest}  ({len(crops)}/{len(CATEGORIES)} regions)")


def main():
    print_data_report()

    args = sys.argv[1:]
    if args:
        photos = [Path(a) for a in args]
    else:
        photos = [p for p in sorted(QUERIES_DIR.glob("*")) if p.suffix.lower() in IMG_EXT]

    if not photos:
        print(f"No query photos found in {QUERIES_DIR} and none passed as arguments.")
        print("Drop a face photo into queries/ and re-run.")
        return

    for p in photos:
        preview_crops(p, OUT_DIR)


if __name__ == "__main__":
    main()
