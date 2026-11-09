"""Text rendering of one photo's matches."""
from avatar_creator.domain.models.avatar_selection import AvatarSelection


class MatchReport:
    """Formats an ``AvatarSelection`` as the console table (category / top-1 / score / runners-up)."""

    def render(self, selection: AvatarSelection) -> str:
        lines = [
            selection.photo_path.name,
            f"  {'category':<12}{'top-1':<20}{'score':>8}   runners-up",
        ]
        for region, hits in selection.matches.items():
            if not hits:
                # A region whose bank ended up empty has nothing to rank.
                lines.append(f"  {region.value:<12}{'(no pieces indexed)':<20}")
                continue
            top = hits[0]
            runners_up = ", ".join(f"{m.piece_id}({m.score:.2f})" for m in hits[1:])
            lines.append(f"  {region.value:<12}{top.piece_id:<20}{top.score:>8.3f}   {runners_up}")
        return "\n".join(lines)
