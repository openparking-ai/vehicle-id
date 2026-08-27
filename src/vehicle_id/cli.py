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
from .engine import RECOMMENDED_CONFIDENCE_THRESHOLD, PlateEngine
from .plates.recognizer import DEFAULT_WEIGHTS

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--threshold",
        type=float,
        default=RECOMMENDED_CONFIDENCE_THRESHOLD,
        help="operating point; the default is the MEASURED one (scripts/eval_plates.py)",
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
    verdict = "ANSWER  " if read.is_answer else "FALLBACK"
    plate = identity.plate or "—"
    print(f"  {verdict}  {plate:<12} confidence {read.confidence:.3f}  {source}")
    if not read.is_answer:
        # Said out loud every time, because this is the property the product is
        # sold on and the one an evaluator is most likely to misread as a bug.
        print(
            f"            below the measured operating point "
            f"({read.threshold_applied:.3f}) — not an error, an answer"
        )


def cmd_read(args) -> int:
    folder = args.folder
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2
    images = _images(folder)
    if not images:
        print(f"no images in {folder}", file=sys.stderr)
        return 2

    engine = PlateEngine(args.weights, device=args.device, threshold=args.threshold)

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

    engine = PlateEngine(args.weights, device=args.device, threshold=args.threshold)
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
