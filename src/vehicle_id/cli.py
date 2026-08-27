"""The command line, for someone evaluating this with no system at all.

That is the whole reason it exists. Anyone can be told a Vehicle ID engine is
good; `vehicle-id read ./photos` is how they find out for themselves in a
minute, on their own images, with no parking system, no database and no
integration.

    vehicle-id read ./photos            reads a folder, prints one record each
    vehicle-id read ./photos --json     the same records, machine-readable
    vehicle-id serve                    the local service on 127.0.0.1:8088

`serve --push-to URL` turns on push delivery to a consumer.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .contract import Capture
from .engine import PlateEngine, UnmeasuredWeights
from .plates.recognizer import DEFAULT_WEIGHTS
from .presence import (
    DEFAULT_MIN_OCCUPANCY,
    KNOWN_LIMITS,
    UNVALIDATED,
    PresenceDetector,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--empty-lane",
        type=Path,
        help="an image of the lane with nothing in it. Without one, presence is "
             "reported as NOT MEASURED and nothing else changes",
    )
    parser.add_argument(
        "--min-occupancy",
        type=float,
        default=DEFAULT_MIN_OCCUPANCY,
        help="how much of the frame a vehicle is expected to fill. An assumption, "
             "not a measurement -- it cannot be measured without lane footage",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the operating point. By default the engine uses the one "
             "MEASURED for these exact weights and refuses to start if there is "
             "not one -- see scripts/eval_plates.py --write-operating-point",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vehicle-id", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    read = sub.add_parser("read", help="read every image in a folder")
    read.add_argument("folder", type=Path)
    read.add_argument("--json", action="store_true", help="emit the records as JSON")
    read.add_argument(
        "--per-vehicle",
        action="store_true",
        help="treat the whole folder as several captures of ONE vehicle",
    )
    _add_engine_args(read)

    serve = sub.add_parser("serve", help="run the local service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8088)
    serve.add_argument("--push-to", help="POST every read to this URL as it happens")
    serve.add_argument("--queue", type=Path, default=Path("var/push-queue.jsonl"))
    _add_engine_args(serve)

    return parser


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _print_human(read, source: str) -> None:
    identity = read.identity
    if read.presence is False:
        print(f"  NO VEHICLE          —            {source}")
        return
    verdict = "ANSWER  " if read.is_answer else "FALLBACK"
    plate = identity.plate or "—"
    seen = f" [{read.captures_seen} captures]" if read.captures_seen > 1 else ""
    print(f"  {verdict}  {plate:<12} confidence {read.confidence:.3f}  {source}{seen}")
    if not read.is_answer:
        # Said out loud every time, because this is the property the product is
        # sold on and the one an evaluator is most likely to misread as a bug.
        print(
            f"            below the measured operating point "
            f"({read.threshold_applied:.3f}) — not an error, an answer"
        )


def _presence(args):
    if not args.empty_lane:
        return None
    import cv2

    reference = cv2.imread(str(args.empty_lane))
    if reference is None:
        print(f"could not read {args.empty_lane}", file=sys.stderr)
        raise SystemExit(2)
    # Said here because this is the moment somebody chooses it. The contract and
    # the README say the same thing, and neither is read by the person typing
    # the flag at 6am. Presence is off by default precisely so that turning it
    # on is a decision, and a decision nobody was told about is not one.
    #
    # The text is `presence.UNVALIDATED` and `presence.KNOWN_LIMITS` rather than
    # a sentence written here, so that this seam and the health endpoint cannot
    # end up disclosing different things.
    print(f"presence gate ON. {UNVALIDATED}", file=sys.stderr)
    for limit in KNOWN_LIMITS:
        print(f"  - {limit}", file=sys.stderr)
    print("  See the presence section of README.md for the measured tables.", file=sys.stderr)
    return PresenceDetector(reference=reference, min_occupancy=args.min_occupancy)


def _engine(args):
    try:
        return PlateEngine(
            args.weights,
            device=args.device,
            threshold=args.threshold,
            presence=_presence(args),
        )
    except UnmeasuredWeights as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return None


def cmd_read(args) -> int:
    folder = args.folder
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2
    images = _images(folder)
    if not images:
        print(f"no images in {folder}", file=sys.stderr)
        return 2

    engine = _engine(args)
    if engine is None:
        return 2

    if args.per_vehicle:
        captures = [Capture.now(p.read_bytes(), p.name) for p in images]
        reads = [(engine.read(captures), f"{len(images)} captures")]
    else:
        reads = [
            (engine.read([Capture.now(p.read_bytes(), p.name)]), str(p.relative_to(folder)))
            for p in images
        ]

    if args.json:
        print(json.dumps([r.to_dict() for r, _ in reads], indent=2))
        return 0

    print(f"\n {engine.engine.name} {engine.engine.version}")
    print(f" weights {engine.engine.weights_id}   operating point {engine.threshold:.3f}\n")
    for read, source in reads:
        _print_human(read, source)
    answered = sum(1 for r, _ in reads if r.is_answer)
    print(f"\n {answered}/{len(reads)} answered, {len(reads) - answered} to fallback\n")
    return 0


def cmd_serve(args) -> int:
    from .push import ReadPusher
    from .service import VehicleIdService, make_server

    engine = _engine(args)
    if engine is None:
        return 2
    pusher = ReadPusher(args.push_to, args.queue) if args.push_to else None
    service = VehicleIdService(engine, pusher=pusher)
    server = make_server(service, host=args.host, port=args.port)

    print(f"vehicle-id on http://{args.host}:{args.port}  (local only by design)")
    print(f"  engine {engine.engine.name} {engine.engine.version}")
    print(f"  weights {engine.engine.weights_id}   operating point {engine.threshold:.3f}")
    if pusher:
        print(f"  pushing every read to {args.push_to}, queued at {args.queue}")
        # Delivers anything left outstanding by a previous run BEFORE the first
        # vehicle of the day arrives, and keeps retrying on its own timer
        # afterwards. Without this, retry is coupled to new traffic.
        pusher.start()
        print(f"  {pusher.stats.pending} read(s) still outstanding from a previous run")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        if pusher:
            pusher.stop()
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    return {"read": cmd_read, "serve": cmd_serve}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
