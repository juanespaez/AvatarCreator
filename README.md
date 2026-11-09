# AvatarCreator

correr python scripts\match_pieces.py curry

1. Abre PowerShell y ve a la carpeta del proyecto:

cd "C:\Users\USER\dev\Nueva carpeta\AvatarCreator"

2. Copia tu foto nueva a algún lugar fácil de escribir, por ejemplo directo a la carpeta queries del proyecto (así ni tienes que escribir la ruta completa):

copy "C:\Users\USER\Downloads\miFoto.jpg" queries\

3. Corre el comando, apuntando a esa foto:

.venv\Scripts\python.exe scripts\match_pieces.py queries\miFoto.jpg
python scripts\match_pieces.py queries\miFoto.jpg
python scripts\match_pieces.py "queries\mi_foto_nueva.jpg"
start "_cache\previews\curry_match.png"

Turns a photo of a face into a Lego-style avatar by matching each facial region (eyes, eyebrows,
nose, mouth, face shape, hair, beard, skin tone) against a bank of Lego piece artwork, using
landmark-based crops, learned embeddings, and per-category vector search.

The original implementation and measurement rig is the Colab notebook
[`scripts/Pipeline.ipynb`](scripts/Pipeline.ipynb). The same pipeline now also exists as an
object-oriented package in [`src/avatar_creator`](src/avatar_creator) (see
[OOP package](#oop-package-src) below); the notebook is kept as the reference it was ported from.

## How it works

1. **MediaPipe Face Mesh** (468/478 points) finds facial landmarks in the input photo and crops
   each region — eyes, eyebrows, nose, mouth, face shape, hair, beard, skin tone — from a
   roll-aligned image, with crop padding scaled to inter-ocular distance so crops are comparable
   across photos of any resolution or framing.
2. Each crop is turned into a vector:
   - **ArcFace** (`buffalo_l`, 512-d) for geometric regions — eyes, eyebrows, nose, mouth, face
     shape.
   - **CLIP** (ViT-B/32, 512-d) for appearance/texture regions — hair, beard.
   - **Median CIELAB + ΔE2000** for skin tone — a 3-number colour metric instead of a neural
     embedding, since skin tone doesn't benefit from (and can't be debugged through) a 512-d space.
3. **FAISS** (`IndexFlatIP`, exact cosine search) finds the nearest Lego piece *within its own
   category* — one index per category, so a beard can never win an eye query.

The resulting per-category top-1 picks are composited into a preview avatar.

## The domain-gap problem, and why `refs/` exists

The query is a real photo; the Lego piece bank is drawn artwork. Photo↔drawing similarity is
dominated by "photo vs. cartoon," not by the actual shape being matched. To remove this gap, the
pipeline supports a **reference set**: real photos labelled with the piece they correspond to.
Matching then runs photo→photo (query against references), and the piece is read off the winning
reference's label — the query and reference embeddings come from an identical crop/align
transform, so the comparison is apples-to-apples. Pure photo→piece matching is kept as a fallback
(`MATCH_MODE = "direct"`) so the difference can be measured rather than assumed.

## Data contract

Assets are read from a Google Drive folder (`ROOT`, edited in cell 4) with this layout:

```
ROOT/
├── pieces/                     # the Lego bank — what gets retrieved
│   ├── eyes/eye_001.png
│   ├── eyebrows/brow_014.png
│   ├── nose/…  mouth/…  face_shape/…  hair/…  beard/…
├── refs/                       # optional but strongly recommended — see domain-gap note above
│   ├── eyes/eye_001/person_a.jpg     # real photos of people whose eyes == piece eye_001
│   ├── eyes/eye_001/person_b.jpg
│   └── eyes/eye_007/person_c.jpg
└── queries/                    # full-face photos to run the pipeline on
    └── test_01.jpg
```

The `refs/<category>/<piece_id>/` folder name *is* the label — no CSV or annotation tool needed.
Adding a piece means dropping a PNG in `pieces/` and, ideally, a few real photos in `refs/`;
nothing is retrained, the index is just rebuilt.

Piece artwork, reference photos, and query photos are all `.gitignore`d — only the folder
structure (via `.gitkeep`) is tracked in git. `refs/` in particular holds real photos of real
people and should never end up in the repo's history.

## Current asset status

The `pieces/` bank was seeded from a scraped asset dump that turned out to be **three overlapping
art-style packs** (Apple, Avatoon, Snapchat), each with partial and inconsistently-gendered
category coverage. Two scoping decisions were made to organize it:

- **Styles are merged, not kept separate.** Apple/Avatoon/Snapchat pieces for the same category
  all compete in one flat pool (e.g. `pieces/eyes/`) rather than being indexed per style. The
  original style (and gender, where the source had one) is preserved in the filename prefix for
  traceability — `apple_01.png`, `avatoon_man_03.png`, `snapchat_woman_07.png` — but matching
  treats them as one undifferentiated set. A composited avatar can therefore currently mix an
  Apple-style nose with a Snapchat-style mouth; there's no style-consistency constraint yet.
- **Only the notebook's original 8 categories are active.** `face_shape`, `eyes`, `eyebrows`,
  `nose`, `mouth`, `beard`, `hair` are populated (counts below); `skin_tone` has **no piece
  artwork yet** — it needs colour swatches, not scraped avatar art, and none were in the source
  dump. Seven more categories from the same dump — **ears, teeth, eyelashes ("Pestañas"), age,
  forehead marks, moles ("Lunares"), freckles ("Pecas")** — are parked under `parked/<slug>/`
  rather than dropped, because the notebook has no `REGION_IDX`/`CROP_CFG`/`ENCODER` entries for
  them yet. **Scaling to these is a known next step**, not a rejection — wiring one in means adding
  a landmark index group, a crop config, and an encoder assignment in cells 6–7, the same way the
  8 active categories are defined.

| category | pieces |
|---|---|
| eyebrows | 33 |
| mouth | 33 |
| eyes | 32 |
| hair | 32 |
| beard | 25 |
| face_shape | 12 |
| nose | 11 |
| skin_tone | 0 — needs swatch artwork |

## Notebook structure

| Cell | Purpose |
|---|---|
| 1 | Runtime & pinned installs (`insightface`, `onnxruntime-gpu`, `open_clip_torch`, `faiss-cpu`, `mediapipe`, etc.) |
| 2 | Mounts Drive, defines `ROOT`/categories, scans `pieces/` and `refs/`, flags orphaned refs |
| 3 | MediaPipe Face Mesh landmark extraction |
| 4 | Roll-aligned, IOD-scaled square region crops |
| 5 | Visual sanity check — run before building any index |
| 6 | ArcFace / CLIP / skin-tone (CIELAB) encoders |
| 7 | Routing table (`ENCODER`) and match mode (`reference` / `direct` / `hybrid`) |
| 8 | Builds one FAISS index per category, cached to Drive |
| 9 | `search_piece` / `build_avatar` — photo → piece per category |
| 10 | Leave-one-out A/B eval of ArcFace vs. CLIP per category, to validate (or flip) the routing table |
| 11 | Composite preview (`canvas` or `anchored` layout) |
| 12 | Persists indices, vectors, ids, and the manifest/config to Drive |

## Running it

1. Open `scripts/Pipeline.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU** (CPU works, but indexing a large bank in cell 8
   will be slow).
3. Edit `ROOT` in cell 4 to point at your Drive folder, laid out as above.
4. Run cells 1–5 and **look at the crops** before doing anything else — if they're wrong,
   everything downstream is meaningless. `CROP_CFG` (cell 8) is the first thing to tune, especially
   the `hair`/`beard` offsets.
5. Run cell 8 with `MATCH_MODE = "direct"` for a baseline, then cell 9 on a few photos. Expect this
   to look mediocre — that's the domain gap, not a bug.
6. Populate `refs/<category>/<piece_id>/` with real photos and run cell 10. Two refs per piece is
   the minimum for a number to exist; five is where it starts to mean something.
7. Set `ENCODER` (cell 7) from cell 10's measured routing, not from assumption.
8. Run cell 12 to persist artifacts (`manifest.json`, `*.vectors.npy`, `*.ids.json`, `*.faiss`).

## OOP package (`src/`)

Layered (DDD) so dependencies only point inwards: `presentation → application → domain ←
ports ← infrastructure`.

```
src/avatar_creator/
├── domain/            pure rules: FaceRegion enum, RegionSpec, crop geometry, PieceRanker
├── ports/             contracts: LandmarkDetector, RegionEncoder, SimilaritySearcher, AssetRepository, ...
├── application/       AvatarRecognizer (Facade), one RegionRecognizer per region,
│                      RecognizerRegistry (enum-keyed), EmbeddingBankBuilder, AvatarCompositor, RoutingEvaluator
├── infrastructure/    MediaPipe, ArcFace, CLIP, FAISS, deltaE, pickle cache, filesystem, matplotlib
└── presentation/      CLI + PipelineContainer (composition root, the only place that knows every class)
```

Each region (`EyeRecognizer`, `EyebrowsRecognizer`, `NoseRecognizer`, `MouthRecognizer`,
`FaceShapeRecognizer`, `HairRecognizer`, `BeardRecognizer`, `SkinToneRecognizer`) is a few lines: it
declares its `RegionSpec` (landmarks, crop framing, encoder) and inherits everything else from
`RegionRecognizer`. **Adding a region** = one `FaceRegion` member + one recognizer subclass + one line
in `RecognizerRegistry`.

```powershell
pip install -e . --no-deps                       # once, inside the venv (deps are in requirements-local.txt)
python -m avatar_creator preview-crops curry     # crop grid (check this first)
python -m avatar_creator match curry drake       # top matches per region (--mode direct|reference|hybrid)
python -m avatar_creator avatar curry            # + assembled avatar image
python -m avatar_creator evaluate                # leave-one-out ArcFace vs CLIP (needs >=2 refs/piece)
python -m avatar_creator export                  # vectors / FAISS indices / manifest -> artifacts/
python -m unittest discover -s tests -t .        # unit tests (no models needed)
```

Results are identical to `scripts/match_pieces.py` (same ids and scores, direct and reference
mode). `scripts/preview_crops.py` and `scripts/match_pieces.py` are kept untouched as that baseline.

**Cache gotcha:** `_cache/*.pkl` is keyed only by region/encoder/mode, not by the piece files or
`RegionSpec`. If you add/remove pieces or tune a crop config, delete `_cache/` (or pass
`--cache-dir`), otherwise results silently use stale vectors.

## Open question / next step

The notebook is the measurement rig, not the service. Porting to FastAPI should be a thin wrapper
around `crops_from_photo` + `search_piece`, loading the cell 12 artifacts at startup. Unresolved:
does each category get an independent argmax (current behaviour, and what's simplest), or does the
avatar need global coherence — e.g. hair colour constraining eyebrow colour? Independent argmax is
simpler; joint selection is a considerably harder problem.
