"""
Local, Colab-free mirror of Pipeline.ipynb cells 6-9: embed either the
pieces/ bank (direct mode) or the refs/ bank (reference mode) with
ArcFace/CLIP, build a FAISS index per category, and match a query photo
against it.

Usage (a photo is a path, or just a name from queries/, extension optional):
    .venv/bin/python scripts/match_pieces.py <photo1> [photo2 ...]
    .venv/bin/python scripts/match_pieces.py --mode reference <photo1> [photo2 ...]
"""
import os

# mediapipe (TFLite/XNNPACK), onnxruntime (ArcFace), and torch (CLIP) each
# bundle their own OpenMP runtime. Loading all three in one process reliably
# segfaults (exit 139) partway through CLIP loading on macOS unless this is
# set before any of them import - must happen before the imports below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preview_crops as pc  # noqa: E402  (reuses landmarks/crop code, already validated)

CACHE_DIR = pc.ROOT / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENCODER = {
    "eyes": "arcface",
    "eyebrows": "arcface",
    "nose": "arcface",
    "mouth": "arcface",
    "face_shape": "arcface",
    "hair": "clip",
    "beard": "clip",
    "skin_tone": "skin",
}

DEVICE = "cpu"


# ---- encoders (mirrors Pipeline.ipynb cell 6) -------------------------------
def _load_arcface():
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    print("Downloading/loading ArcFace (buffalo_l) ...", flush=True)
    _ = FaceAnalysis(name="buffalo_l")
    arc_path = str(Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx")
    model = get_model(arc_path)
    model.prepare(ctx_id=-1)
    print("ArcFace ready.", flush=True)
    return model


def _load_clip():
    import open_clip
    import torch

    print("Downloading/loading CLIP (ViT-B-32, laion2b_s34b_b79k) ...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(DEVICE).eval()
    print("CLIP ready.", flush=True)
    return model, preprocess


_ARC = None
_CLIP_MODEL = None
_CLIP_PRE = None


def embed_arcface(crops_rgb):
    global _ARC
    if _ARC is None:
        _ARC = _load_arcface()
    batch = []
    for c in crops_rgb:
        img = cv2.resize(c, (112, 112), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = (img.astype(np.float32) - 127.5) / 127.5
        batch.append(img.transpose(2, 0, 1))
    blob = np.stack(batch).astype(np.float32)
    feats = _ARC.session.run(None, {_ARC.session.get_inputs()[0].name: blob})[0]
    return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)


def embed_clip(crops_rgb, batch_size=64):
    import torch
    from PIL import Image

    global _CLIP_MODEL, _CLIP_PRE
    if _CLIP_MODEL is None:
        _CLIP_MODEL, _CLIP_PRE = _load_clip()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops_rgb), batch_size):
            t = torch.stack(
                [_CLIP_PRE(Image.fromarray(c)) for c in crops_rgb[i : i + batch_size]]
            ).to(DEVICE)
            f = _CLIP_MODEL.encode_image(t).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def embed_skin(crops_rgb):
    from skimage import color as skcolor

    out = []
    for c in crops_rgb:
        lab = skcolor.rgb2lab(c.astype(np.float32) / 255.0)
        out.append(np.median(lab.reshape(-1, 3), axis=0))
    return np.asarray(out, dtype=np.float32)


def skin_distance(q, bank):
    from skimage.color import deltaE_ciede2000

    return np.array([deltaE_ciede2000(q, b) for b in bank], dtype=np.float32)


def embed(crops_rgb, encoder):
    if not len(crops_rgb):
        return np.zeros((0, 512), np.float32)
    return {"arcface": embed_arcface, "clip": embed_clip, "skin": embed_skin}[encoder](crops_rgb)


# ---- piece-image prep (mirrors Pipeline.ipynb cell 8) -----------------------
def _prep_piece_image(path, out_size=224):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3 and raw.shape[2] == 4:
        a = raw[:, :, 3:4].astype(np.float32) / 255.0
        rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        raw = (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
    else:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    h, w = raw.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 255, np.uint8)
    canvas[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = raw
    return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_CUBIC)


def _make_index(vecs, encoder):
    import faiss

    if encoder == "skin" or len(vecs) == 0:
        return None
    ix = faiss.IndexFlatIP(vecs.shape[1])
    ix.add(vecs.astype(np.float32))
    return ix


def build_category(category, mode, items):
    import pickle

    enc = ENCODER[category]
    cache = CACHE_DIR / f"{category}__{enc}__{mode}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            d = pickle.load(f)
        d["index"] = _make_index(d["vectors"], enc)
        return d

    if mode == "direct":
        get_crop = _prep_piece_image
    else:
        get_crop = lambda path: pc.crops_from_photo(path, categories=[category]).get(category)  # noqa: E731

    ids, crops, failed = [], [], 0
    for piece_id, path in items:
        img = get_crop(path)
        if img is None:
            failed += 1
            continue
        ids.append(piece_id)
        crops.append(img)

    vecs = embed(crops, enc) if crops else np.zeros((0, 512), np.float32)
    d = dict(ids=ids, vectors=vecs, mode=mode, encoder=enc, n_failed=failed)
    with open(cache, "wb") as f:
        pickle.dump(d, f)
    d["index"] = _make_index(vecs, enc)
    return d


def search_piece(crop_rgb, category, indexes, k=3):
    d = indexes[category]
    enc = d["encoder"]
    q = embed([crop_rgb], enc)[0]

    if enc == "skin":
        dist = skin_distance(q, d["vectors"])
        order = np.argsort(dist)
        agg = {}
        for i in order:
            agg.setdefault(d["ids"][i], -float(dist[i]))
        return sorted(agg.items(), key=lambda kv: -kv[1])[:k]

    n_fetch = min(len(d["ids"]), max(k * 10, 50))
    sims, idxs = d["index"].search(q.reshape(1, -1).astype(np.float32), n_fetch)
    best = defaultdict(lambda: -1e9)
    for s, i in zip(sims[0], idxs[0]):
        if i < 0:
            continue
        pid = d["ids"][i]
        best[pid] = max(best[pid], float(s))
    return sorted(best.items(), key=lambda kv: -kv[1])[:k]


def build_avatar(photo_path, indexes, bank, k=3, preview_dir=None):
    crops = pc.crops_from_photo(photo_path)
    if not crops:
        print(f"NO FACE DETECTED: {photo_path}")
        return None
    result = {}
    for cat in pc.CATEGORIES:
        if cat not in indexes or cat not in crops:
            continue
        result[cat] = search_piece(crops[cat], cat, indexes, k=k)
    print(Path(photo_path).name)
    print(f"  {'category':<12}{'top-1':<20}{'score':>8}   runners-up")
    for cat, hits in result.items():
        if not hits:
            print(f"  {cat:<12}{'(no pieces indexed)':<20}")
            continue
        top, sc = hits[0]
        ru = ", ".join(f"{p}({s:.2f})" for p, s in hits[1:])
        print(f"  {cat:<12}{top:<20}{sc:>8.3f}   {ru}")
    if preview_dir is not None:
        save_match_preview(photo_path, result, crops, bank, preview_dir)
    return result


def save_match_preview(photo_path, result, crops, bank, out_dir):
    """Top row: the query crop per category. Bottom row: the top-1 matched
    piece, so a match can be eyeballed for plausibility rather than trusted
    on cosine score alone."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = [c for c in pc.CATEGORIES if c in result and result[c]]
    if not cats:
        return
    path_of = {cat: dict(bank[cat]) for cat in cats}

    fig, axes = plt.subplots(2, len(cats), figsize=(2.2 * len(cats), 5.0))
    if len(cats) == 1:
        axes = axes.reshape(2, 1)

    for col, cat in enumerate(cats):
        axes[0, col].imshow(crops[cat])
        axes[0, col].set_title(cat, fontsize=9)
        axes[0, col].axis("off")

        pid, score = result[cat][0]
        piece_path = path_of[cat].get(pid)
        piece_img = _prep_piece_image(piece_path, out_size=224) if piece_path else None
        if piece_img is not None:
            axes[1, col].imshow(piece_img)
        axes[1, col].set_title(f"{pid}\n{score:.2f}", fontsize=8)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("query", fontsize=9)
    axes[1, 0].set_ylabel("top-1 match", fontsize=9)
    fig.suptitle(Path(photo_path).name, fontsize=11)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{Path(photo_path).stem}_match.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"  -> saved comparison: {dest}")


def resolve_query(name):
    """Accept a full path, or just a file name (extension optional) inside queries/."""
    p = Path(name)
    if p.exists():
        return p
    candidates = [pc.QUERIES_DIR / name] + [
        pc.QUERIES_DIR / f"{name}{ext}" for ext in sorted(pc.IMG_EXT)
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"No image {name!r} found (also looked in {pc.QUERIES_DIR})")


def main():
    args = sys.argv[1:]
    mode = "direct"
    if "--mode" in args:
        i = args.index("--mode")
        mode = args[i + 1]
        del args[i : i + 2]
    if mode not in ("direct", "reference"):
        print(f"Unknown --mode {mode!r}, expected 'direct' or 'reference'")
        return
    if not args:
        print("Usage: match_pieces.py [--mode direct|reference] <photo1> [photo2 ...]")
        return

    bank = pc.scan_bank()
    source = bank if mode == "direct" else pc.scan_refs()
    noun = "pieces" if mode == "direct" else "refs"
    indexes = {}
    for cat in pc.CATEGORIES:
        if not source[cat]:
            print(f"skip {cat}: no {noun}")
            continue
        print(f"building index: {cat} ({ENCODER[cat]}, {len(source[cat])} {noun}, mode={mode}) ...", flush=True)
        indexes[cat] = build_category(cat, mode, source[cat])
        d = indexes[cat]
        print(f"  -> {len(d['ids'])} vectors, {d['n_failed']} failed")

    preview_dir = pc.ROOT / "_cache" / "previews"
    print()
    for photo in args:
        build_avatar(resolve_query(photo), indexes, bank, preview_dir=preview_dir)
        print()


if __name__ == "__main__":
    main()
