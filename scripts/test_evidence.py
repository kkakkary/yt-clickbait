"""Run the SPEC test suites and write a per-case evidence index."""

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
MODEL_EVIDENCE = {
    "FT-3": "Held-out model metrics and clustered OLS coefficient tables",
    "FT-6": "Three-target embedding comparison on both splits",
    "FT-7": "Cluster report and human review",
}


def run_suite(output, name, args):
    xml = output / f"{name}.xml"
    xml.unlink(missing_ok=True)
    command = [
        sys.executable, "-m", "pytest", "-ra", "--continue-on-collection-errors",
        *args, f"--junitxml={xml}",
    ]
    started = perf_counter()
    with (output / f"{name}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    elapsed = perf_counter() - started
    cases = {}
    if xml.exists():
        for case in ET.parse(xml).iter("testcase"):
            key = (case.get("classname").rsplit(".", 1)[-1], case.get("name").split("[")[0])
            status = next(
                (
                    label for label in ("error", "failure", "skipped")
                    if case.find(label) is not None
                ),
                "passed",
            )
            cases.setdefault(key, []).append(status)
    status = "passed" if result.returncode == 0 else f"failed (exit {result.returncode})"
    return status, elapsed, cases


def case_status(cases, file, function):
    statuses = cases.get((Path(file).stem, function), [])
    return ", ".join(sorted(set(statuses))) if statuses else "not run"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "test-evidence")
    parser.add_argument("--full", action="store_true", help="Also run model-download tests")
    parser.add_argument("--pipeline", action="store_true", help="Run FT-1 and FT-2 on raw data")
    parser.add_argument(
        "--model-evidence", action="append", default=[], metavar="FT-ID=PATH",
        help="Link an existing FT-3, FT-6, or FT-7 report for review; repeat for multiple files",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    model_files = {case_id: [] for case_id in MODEL_EVIDENCE}
    for item in args.model_evidence:
        case_id, sep, filename = item.partition("=")
        if not sep or case_id not in model_files or not Path(filename).is_file():
            parser.error(f"expected FT-3, FT-6, or FT-7 and an existing file: {item}")
        model_files[case_id].append(Path(filename).resolve())
    output.mkdir(parents=True, exist_ok=True)
    fast, elapsed, cases = run_suite(output, "fast", ["-m", "not slow and not full_data"])
    full = "not run"
    if args.full:
        full, seconds, full_cases = run_suite(output, "full", ["-m", "not full_data"])
        full += f" ({seconds:.2f} s)"
        cases.update(full_cases)
    pipeline = "not run"
    pipeline_cases = {}
    if args.pipeline:
        pipeline, _, pipeline_cases = run_suite(
            output, "pipeline", ["-m", "full_data", "--capture=tee-sys"]
        )

    lines = [
        "# SPEC test evidence",
        "",
        "| Test | Result | Evidence |",
        "|---|---|---|",
    ]
    spec = (ROOT / "docs" / "SPEC.md").read_text()
    unit_rows = re.findall(r"^\|\s*★?(UT-\d+)\s*\|\s*`([^:`]+)::([^`]+)`", spec, re.M)
    for case_id, file, function in unit_rows:
        status = case_status(cases, file, function)
        suite = "full" if args.full else "fast"
        lines.append(
            f"| {case_id} | {status} | `{file}::{function}` · [{suite} log]({suite}.log) |"
        )
    for case_id, function in [("FT-1", "test_build_dataset"), ("FT-2", "test_build_features")]:
        status = case_status(pipeline_cases, "test_spec_pipeline.py", function)
        evidence = "[Pipeline log](pipeline.log)" if args.pipeline else "Run with `--pipeline`"
        lines.append(f"| {case_id} | {status} | {evidence} |")
    within_budget = "unverified"
    if fast == "passed":
        within_budget = "met" if elapsed < 10 else "exceeded"
    full_log = "[Full log](full.log)" if args.full else "Run with `--full`"
    lines.extend([
        f"| FT-4 | {fast}, {elapsed:.2f} s, 10 s budget {within_budget} | [Fast log](fast.log) |",
        f"| FT-5 | {full} | {full_log} |",
    ])
    for case_id, description in MODEL_EVIDENCE.items():
        files = model_files[case_id]
        if files:
            links = ", ".join(f"[{path.name}](<{os.path.relpath(path, output)}>)" for path in files)
            lines.append(f"| {case_id} | attached for review | {links} |")
        else:
            lines.append(f"| {case_id} | evidence pending | {description} |")
    index = output / "index.md"
    index.write_text("\n".join(lines) + "\n")
    print(index)
    return int(any(result.startswith("failed") for result in (fast, full, pipeline)))


if __name__ == "__main__":
    raise SystemExit(main())
