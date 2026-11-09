"""What a photo query is compared against."""
from enum import Enum


class MatchMode(Enum):
    """Which bank the query is searched in.

    * ``REFERENCE`` - real photos labelled with a piece id (photo <-> photo, no domain gap).
    * ``DIRECT``    - the piece artwork itself (photo <-> drawing, dominated by rendering style).
    * ``HYBRID``    - reference when a region has enough labelled refs, otherwise direct.
                      It is only a *policy*: it is always resolved to DIRECT or REFERENCE
                      per region (see ``resolve``) before anything is built.
    """

    DIRECT = "direct"
    REFERENCE = "reference"
    HYBRID = "hybrid"

    def resolve(self, n_refs: int, min_refs: int) -> "MatchMode":
        """Return the concrete mode (DIRECT or REFERENCE) to use for one region.

        Fewer than ``min_refs`` labelled photos is noise rather than signal, which is why
        HYBRID falls back to DIRECT below that threshold.
        """
        if self is not MatchMode.HYBRID:
            return self
        return MatchMode.REFERENCE if n_refs >= min_refs else MatchMode.DIRECT
