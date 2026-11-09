"""AvatarCreator: turn a face photo into a Lego-style avatar.

Package layout (dependencies point strictly inwards, towards ``domain``)::

    presentation  ->  application  ->  domain  <-  ports  <-  infrastructure
       (CLI)        (use cases)      (rules)    (contracts)  (cv2/torch/faiss...)

* ``domain``         - pure concepts and rules: regions, crop geometry, ranking.
* ``ports``          - abstract contracts the application needs from the outside world.
* ``application``    - use cases: recognizers, the AvatarRecognizer facade, evaluation, compositing.
* ``infrastructure`` - concrete adapters: MediaPipe, ArcFace, CLIP, FAISS, filesystem, matplotlib.
* ``presentation``   - the command line and the composition root that wires everything together.
"""
import os

# WHY here: mediapipe (TFLite/XNNPACK), onnxruntime (ArcFace) and torch (CLIP) each ship their
# own OpenMP runtime. Loading all three in one process can segfault (seen on macOS) unless
# duplicate runtimes are tolerated and threading is pinned. These variables only take effect if
# they are set BEFORE any of those libraries is imported, and every entry point of the project
# imports this package first - so this is the one safe place to set them.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
