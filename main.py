r"""One-click launcher: paste a photo URL below, run this file, get the avatar.

    .venv\Scripts\python.exe main.py                      # uses PHOTO below
    .venv\Scripts\python.exe main.py https://site/me.jpg  # or pass the link on the command line

PHOTO can also be a local path or the name of a file in queries/ (e.g. "curry").
The URL must point straight at the image file (jpeg, png or webp), not at a web page.
Results are written to _cache\previews\: <name>_match.png and <name>_avatar.png.
"""
import sys
from pathlib import Path

# <-- PASTE THE PHOTO URL HERE
PHOTO = "https://example.com/my_face.jpg"

# "direct"    = match against the piece artwork (works with no reference photos)
# "reference" = match against real photos labelled in refs/ (needs refs for every region)
# "hybrid"    = reference where a region has enough refs, direct elsewhere
MODE = "direct"

# Make `avatar_creator` importable without installing it (src/ layout).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from avatar_creator.__main__ import main  # noqa: E402  (must come after the sys.path line)

if __name__ == "__main__":
    # A link given on the command line wins over the PHOTO constant above.
    photo = sys.argv[1] if len(sys.argv) > 1 else PHOTO
    sys.exit(main(["avatar", "--mode", MODE, photo]))
