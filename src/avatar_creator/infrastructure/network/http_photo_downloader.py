"""urllib adapter for the PhotoDownloader port."""
import hashlib
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from avatar_creator.domain.photo_download_error import PhotoDownloadError
from avatar_creator.ports.photo_downloader import PhotoDownloader


class HttpPhotoDownloader(PhotoDownloader):
    """Downloads a photo over http(s) using only the standard library.

    The URL is user-supplied, so it is treated as untrusted input: only http/https is allowed,
    the server must say it is serving an image of a type the pipeline can read, and the body is
    capped in size and time so a wrong link cannot fill the disk or hang the run.
    """

    #: Content types the pipeline can read, mapped to the extension to store them under.
    #: The extension is taken from the content type (not the URL) because URLs often have none.
    EXTENSIONS: dict[str, str] = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }

    def __init__(self, timeout_seconds: float = 20.0, max_bytes: int = 15 * 1024 * 1024) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes

    def download(self, url: str, dest_dir: Path) -> Path:
        scheme = urlparse(url).scheme.lower()
        if scheme not in ("http", "https"):
            raise PhotoDownloadError(f"only http(s) URLs are supported, got {url!r}")

        # Same URL -> same file name, so re-running with one link does not download it again.
        stem = self._safe_stem(url)
        for extension in set(self.EXTENSIONS.values()):
            existing = dest_dir / f"{stem}{extension}"
            if existing.is_file():
                return existing

        # Many image hosts answer 403 to the default urllib client, so identify as a plain app.
        request = urllib.request.Request(url, headers={"User-Agent": "AvatarCreator/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                content_type = response.headers.get_content_type()
                extension = self.EXTENSIONS.get(content_type)
                if extension is None:
                    raise PhotoDownloadError(
                        f"{url} is not a supported image (server says {content_type!r}; "
                        f"need jpeg, png or webp). Use a direct link to the image file."
                    )
                # Read one byte past the cap: if we get it, the file is too big (a Content-Length
                # header is optional and can lie, so the body itself is what gets limited).
                data = response.read(self._max_bytes + 1)
        except urllib.error.HTTPError as error:
            raise PhotoDownloadError(f"{url} answered HTTP {error.code} {error.reason}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise PhotoDownloadError(f"could not download {url}: {error}") from error

        if len(data) > self._max_bytes:
            raise PhotoDownloadError(f"{url} is larger than {self._max_bytes // (1024 * 1024)} MB")

        dest_dir.mkdir(parents=True, exist_ok=True)
        destination = dest_dir / f"{stem}{extension}"
        destination.write_bytes(data)
        return destination

    @staticmethod
    def _safe_stem(url: str) -> str:
        """File name base: the URL's own file name (sanitised) plus a short hash of the full URL.

        The hash keeps different links that share a name (``image.jpg``) from overwriting each
        other; the sanitising keeps a hostile URL from escaping ``dest_dir`` with ``..`` or
        path separators.
        """
        name = Path(urlparse(url).path).stem or "photo"
        name = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:60]
        return f"{name}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]}"
