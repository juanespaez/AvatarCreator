"""Contract: fetch a photo from a URL."""
from abc import ABC, abstractmethod
from pathlib import Path


class PhotoDownloader(ABC):
    """Turns an image URL into a local file the rest of the pipeline can read."""

    @abstractmethod
    def download(self, url: str, dest_dir: Path) -> Path:
        """Save the image at ``url`` inside ``dest_dir`` and return its path.

        Raises:
            PhotoDownloadError: if the URL is unusable or does not serve a supported image.
        """
