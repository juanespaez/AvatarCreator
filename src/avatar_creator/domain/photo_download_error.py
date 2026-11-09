"""Raised when a photo URL cannot be turned into a local image."""


class PhotoDownloadError(RuntimeError):
    """The URL was rejected or the download failed (bad scheme, network error, not an image, too big).

    Always carries a human-readable reason, because the CLI prints it directly to the user.
    """
