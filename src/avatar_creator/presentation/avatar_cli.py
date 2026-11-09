"""Command line interface."""
import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from avatar_creator.domain.models.match_mode import MatchMode
from avatar_creator.domain.no_face_detected_error import NoFaceDetectedError
from avatar_creator.domain.photo_download_error import PhotoDownloadError
from avatar_creator.infrastructure.filesystem.project_paths import ProjectPaths
from avatar_creator.presentation.data_report import DataReport
from avatar_creator.presentation.match_report import MatchReport
from avatar_creator.presentation.pipeline_container import PipelineContainer


class AvatarCli:
    """Parses arguments and runs one command against a freshly wired ``PipelineContainer``.

    It contains no business logic: each command just calls a use case and prints the result.

    Commands:
        preview-crops  Save a crop grid per photo (the check to run before building indices).
        match          Fit the indices and print/save the top matches per region.
        avatar         Like ``match`` but also composites and saves the assembled avatar.
        evaluate       Leave-one-out ArcFace-vs-CLIP measurement (needs >=2 refs per piece).
        export         Persist vectors, indices and manifest to ``artifacts/``.
    """

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Run the CLI and return a process exit code (non-zero if any photo failed)."""
        args = self._build_parser().parse_args(argv)
        self._configure_logging(args.verbose)
        container = PipelineContainer(
            paths=ProjectPaths(root=args.root, cache_dir=args.cache_dir),
            mode=MatchMode(getattr(args, "mode", MatchMode.DIRECT.value)),
            min_refs=getattr(args, "min_refs", 3),
        )
        command = getattr(self, f"_cmd_{args.command.replace('-', '_')}")
        try:
            return command(container, args)
        except (FileNotFoundError, PhotoDownloadError) as error:
            # A mistyped photo name or a bad link is a user error, not a crash: say so and exit non-zero.
            print(f"error: {error}", file=sys.stderr)
            return 2

    # ---- parser --------------------------------------------------------------------------
    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="avatar-creator", description="Photo -> Lego-style avatar.")
        parser.add_argument("--root", type=Path, default=None, help="project root (default: this repository)")
        parser.add_argument("--cache-dir", type=Path, default=None, help="embedding cache (default: <root>/_cache)")
        parser.add_argument("-v", "--verbose", action="store_true", help="also show library debug output")
        commands = parser.add_subparsers(dest="command", required=True)

        preview = commands.add_parser("preview-crops", help="save region crops for each photo")
        preview.add_argument("photos", nargs="*", help="paths or names in queries/ (default: all of queries/)")

        for name, help_text in (("match", "match photos against the piece bank"),
                                ("avatar", "match and assemble the avatar image")):
            command = commands.add_parser(name, help=help_text)
            command.add_argument("photos", nargs="+", help="paths or names in queries/ (extension optional)")
            command.add_argument(
                "--mode", choices=[m.value for m in MatchMode], default=MatchMode.DIRECT.value,
                help="bank to search: direct (pieces), reference (labelled photos) or hybrid",
            )
            command.add_argument("--min-refs", type=int, default=3, help="refs per region HYBRID needs")
            command.add_argument("-k", type=int, default=3, help="matches to keep per region")

        commands.add_parser("evaluate", help="leave-one-out encoder A/B on refs/")
        export = commands.add_parser("export", help="persist artifacts to artifacts/")
        export.add_argument(
            "--mode", choices=[m.value for m in MatchMode], default=MatchMode.DIRECT.value,
            help="mode the banks are built in",
        )
        export.add_argument("--min-refs", type=int, default=3, help="refs per region HYBRID needs")
        return parser

    @staticmethod
    def _configure_logging(verbose: bool) -> None:
        # Only OUR logger prints plain lines to stdout; configuring the root logger would also
        # surface every third-party library's INFO chatter.
        logger = logging.getLogger("avatar_creator")
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)

    @staticmethod
    def _resolve_photo(container: PipelineContainer, name: str) -> Path:
        """A photo argument may be a URL (downloaded into queries/), a path, or a name in queries/."""
        if name.lower().startswith(("http://", "https://")):
            path = container.photo_downloader.download(name, container.paths.queries_dir)
            print(f"downloaded {name} -> {path}")
            return path
        return container.assets.resolve_query(name)

    # ---- commands ------------------------------------------------------------------------
    def _cmd_preview_crops(self, container: PipelineContainer, args: argparse.Namespace) -> int:
        print(DataReport(container.assets).render(), end="\n\n")
        photos = [self._resolve_photo(container, p) for p in args.photos] or container.assets.list_queries()
        if not photos:
            print(f"No query photos found in {container.paths.queries_dir} and none passed as arguments.")
            print("Drop a face photo into queries/ and re-run.")
            return 0

        specs = [r.spec for r in container.avatar_recognizer.recognizers.values()]
        failed = 0
        for photo in photos:
            crops = container.face_cropper.crop_photo(photo, specs)
            if not crops:
                print(f"NO FACE DETECTED: {photo}")
                failed += 1
                continue
            dest = container.crop_preview_renderer.render(photo, crops, container.paths.previews_dir)
            print(f"OK  {photo.name} -> {dest}  ({len(crops)}/{len(specs)} regions)")
        return 1 if failed else 0

    def _cmd_match(self, container: PipelineContainer, args: argparse.Namespace) -> int:
        return self._match(container, args, assemble=False)

    def _cmd_avatar(self, container: PipelineContainer, args: argparse.Namespace) -> int:
        return self._match(container, args, assemble=True)

    def _match(self, container: PipelineContainer, args: argparse.Namespace, assemble: bool) -> int:
        recognizer = container.avatar_recognizer
        recognizer.fit()
        print()
        report, failed = MatchReport(), 0
        for name in args.photos:
            photo = self._resolve_photo(container, name)
            try:
                selection = recognizer.recognize(photo, k=args.k)
            except NoFaceDetectedError:
                print(f"NO FACE DETECTED: {photo}\n")
                failed += 1
                continue
            print(report.render(selection))
            preview = container.match_preview_renderer.render(selection, container.paths.previews_dir)
            if preview:
                print(f"  -> saved comparison: {preview}")
            if assemble:
                avatar = container.compositor.compose(selection)
                dest = container.avatar_preview_renderer.render(
                    photo, container.image_loader.load_photo(photo), avatar, container.paths.previews_dir
                )
                print(f"  -> saved avatar: {dest}")
            print()
        return 1 if failed else 0

    def _cmd_evaluate(self, container: PipelineContainer, args: argparse.Namespace) -> int:
        specs = {region: r.spec for region, r in container.avatar_recognizer.recognizers.items()}
        rows = container.evaluator.evaluate_all(specs)
        if not rows:
            print("not enough labelled refs (need >=2 photos per piece) to evaluate anything yet.")
            return 0

        print(f"{'category':<12}{'encoder':<10}{'n':>5}{'pieces':>8}{'top1':>8}{'top3':>8}{'chance':>8}  verdict")
        for row in rows:
            r = row.result
            verdict = "above chance" if r.above_chance else "NOT above chance"
            print(
                f"{row.region.value:<12}{row.encoder.value:<10}{r.n:>5}{r.n_pieces:>8}"
                f"{r.top1:>8.2f}{r.top3:>8.2f}{r.chance:>8.2f}  {verdict}"
            )

        print("\nmeasured routing (compare with the encoder declared in each recognizer's spec):")
        for region, (encoder, top1) in container.evaluator.best_routing(rows).items():
            declared = specs[region].encoder
            flag = "" if encoder is declared else f"   <-- CHANGED from {declared.value}"
            print(f'    "{region.value}": "{encoder.value}",   # top1={top1:.2f}{flag}')
        return 0

    def _cmd_export(self, container: PipelineContainer, args: argparse.Namespace) -> int:
        recognizer = container.avatar_recognizer
        banks = recognizer.fit()
        specs = {region: r.spec for region, r in recognizer.recognizers.items()}
        counts = {
            region: (len(container.assets.pieces(region)), len(container.assets.references(region)))
            for region in specs
        }
        written = container.exporter.export(banks, specs, counts, recognizer.mode, recognizer.min_refs)
        print(f"written to {container.paths.artifacts_dir}")
        for path in sorted(written):
            print(f"  {path.name:<28}{path.stat().st_size / 1024:>10.1f} KB")
        return 0
