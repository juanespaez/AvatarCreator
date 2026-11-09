"""HttpPhotoDownloader against a throw-away local web server (no internet needed)."""
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from avatar_creator.domain.photo_download_error import PhotoDownloadError
from avatar_creator.infrastructure.network.http_photo_downloader import HttpPhotoDownloader

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64


class _Handler(BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self) -> None:  # noqa: N802  (name required by BaseHTTPRequestHandler)
        type(self).hits += 1
        routes = {
            "/face.png": (200, "image/png", PNG_BYTES),
            "/noext": (200, "image/jpeg", b"jpegdata"),
            "/page.html": (200, "text/html", b"<html></html>"),
            "/big.png": (200, "image/png", b"0" * 5000),
            "/missing.png": (404, "text/plain", b"nope"),
        }
        status, content_type, body = routes.get(self.path, (404, "text/plain", b"nope"))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # keep test output quiet
        pass


class HttpPhotoDownloaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self._tmp.name) / "queries"  # does not exist yet: downloader must create it
        self.downloader = HttpPhotoDownloader(timeout_seconds=5, max_bytes=1000)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_downloads_an_image_into_dest_dir(self) -> None:
        path = self.downloader.download(f"{self.base}/face.png", self.dest)
        self.assertEqual(path.parent, self.dest)
        self.assertEqual(path.suffix, ".png")
        self.assertEqual(path.read_bytes(), PNG_BYTES)

    def test_extension_comes_from_the_content_type_when_the_url_has_none(self) -> None:
        self.assertEqual(self.downloader.download(f"{self.base}/noext", self.dest).suffix, ".jpg")

    def test_same_url_is_downloaded_only_once(self) -> None:
        before = _Handler.hits
        first = self.downloader.download(f"{self.base}/face.png", self.dest)
        second = self.downloader.download(f"{self.base}/face.png", self.dest)
        self.assertEqual(first, second)
        self.assertEqual(_Handler.hits - before, 1)

    def test_rejects_non_image_content(self) -> None:
        with self.assertRaisesRegex(PhotoDownloadError, "not a supported image"):
            self.downloader.download(f"{self.base}/page.html", self.dest)
        self.assertFalse(self.dest.exists() and any(self.dest.iterdir()))  # nothing left behind

    def test_rejects_oversized_bodies(self) -> None:
        with self.assertRaisesRegex(PhotoDownloadError, "larger than"):
            self.downloader.download(f"{self.base}/big.png", self.dest)

    def test_http_errors_become_a_clear_message(self) -> None:
        with self.assertRaisesRegex(PhotoDownloadError, "HTTP 404"):
            self.downloader.download(f"{self.base}/missing.png", self.dest)

    def test_unreachable_host_becomes_a_clear_message(self) -> None:
        with self.assertRaisesRegex(PhotoDownloadError, "could not download"):
            self.downloader.download("http://127.0.0.1:1/x.png", self.dest)

    def test_only_http_and_https_are_allowed(self) -> None:
        for url in ("file:///etc/passwd", "ftp://example.com/a.png", "javascript:alert(1)"):
            with self.assertRaisesRegex(PhotoDownloadError, "only http"):
                self.downloader.download(url, self.dest)

    def test_hostile_file_names_cannot_escape_the_destination(self) -> None:
        stem = HttpPhotoDownloader._safe_stem("http://h/a/..%2F..%2Fevil.png")
        self.assertNotIn("/", stem)
        self.assertNotIn("..", stem)

    def test_different_urls_with_the_same_file_name_do_not_collide(self) -> None:
        a = HttpPhotoDownloader._safe_stem("http://one.example/image.jpg")
        b = HttpPhotoDownloader._safe_stem("http://two.example/image.jpg")
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
