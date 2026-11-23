"""Command line. Human text and diagnostics are separate from result.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from machineear import __version__
from machineear.pipeline import analyze_manifest, validate_manifest
from machineear.report import render_text, write_outputs

CASES = {
    "showcase": "showcase/manifest.csv",
    "recall-loss": "recall_loss/manifest.csv",
    "insufficient": "insufficient/manifest.csv",
    "missing": "missing_metadata/manifest.csv",
    "invalid": "invalid/manifest.csv",
    "overlap": "overlap/manifest.csv",
}


def fixture_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "demo":
        return _demo(args)
    parser.print_help(sys.stderr)
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="machineear",
        description="Audit a frozen machine-audio anomaly threshold after the microphone or background changes.",
    )
    parser.add_argument("--version", action="version", version=f"machineear {__version__}")
    commands = parser.add_subparsers(dest="command")
    validate = commands.add_parser("validate", help="Accept a manifest or reject it with a diagnostic")
    validate.add_argument("--manifest", required=True, type=Path)
    analyze = commands.add_parser("analyze", help="Score clips and write a report")
    analyze.add_argument("--manifest", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--domain", default=None)
    analyze.add_argument("--json", action="store_true", help="Write machine-readable JSON to stdout")
    demo = commands.add_parser("demo", help="Run a packaged fixture through the report")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--case", choices=sorted(CASES), default="showcase")
    demo.add_argument("--domain", default=None)
    demo.add_argument("--json", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> int:
    result = validate_manifest(args.manifest)
    issues = result["issues"]
    if result["status"] == "ok" and not issues:
        print(f"accepted {len(result['clips'])} clips from {args.manifest}")
        return 0
    print(f"rejected {args.manifest}", file=sys.stderr)
    for issue in issues:
        clip = issue.get("clip_id")
        prefix = f"{clip}: " if clip else ""
        print(f"{issue['code']}: {prefix}{issue['message']}", file=sys.stderr)
    print(result["recovery"], file=sys.stderr)
    return 2


def _analyze(args: argparse.Namespace) -> int:
    result = analyze_manifest(args.manifest, domain=args.domain, score=True)
    paths = write_outputs(result, args.output)
    return _emit(result, paths, json_stdout=args.json)


def _demo(args: argparse.Namespace) -> int:
    manifest = fixture_root() / CASES[args.case]
    domain = args.domain
    if args.case == "showcase" and domain is None:
        domain = "mic-b"
    result = analyze_manifest(manifest, domain=domain, score=True)
    paths = write_outputs(result, args.output)
    return _emit(result, paths, json_stdout=args.json)


def _emit(result: dict[str, object], paths: dict[str, Path], *, json_stdout: bool) -> int:
    summary = render_text(result)
    summary = summary + f"\nreport: {paths['html']}"
    if json_stdout:
        print(json.dumps(result, sort_keys=True))
        print(summary, file=sys.stderr)
    else:
        print(summary)
    if result["status"] in {"rejected_input", "missing_metadata"}:
        return 2
    if result["anomaly_status_omitted"] and result["status"] != "ok":
        return 2
    return 0
