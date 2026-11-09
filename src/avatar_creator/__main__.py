"""Entry point: ``python -m avatar_creator <command>`` or the ``avatar-creator`` script."""
import sys
from typing import Sequence

from avatar_creator.presentation.avatar_cli import AvatarCli


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    return AvatarCli().run(argv)


if __name__ == "__main__":
    sys.exit(main())
