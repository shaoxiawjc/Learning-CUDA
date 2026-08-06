#!/usr/bin/env python3

import argparse
import json
import os
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CASE_RE = re.compile(r"Test Case #(\d+) \((rmsNorm|Attention)\)", re.IGNORECASE)
DTYPE_RE = re.compile(r"Data Type:\s*(\w+)")
TIME_RE = re.compile(r"Avg Time:\s*([0-9.eE+-]+)\s*ms")


def run_make(operator):
    command = ["make", "VERBOSE=true"]
    env = os.environ.copy()
    env.pop("SKIP_RMS_NORM", None)
    env.pop("SKIP_ATTENTION", None)
    if operator == "rms_norm":
        env["SKIP_ATTENTION"] = "1"
    elif operator == "attention":
        env["SKIP_RMS_NORM"] = "1"

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
    )
    output = ANSI_RE.sub("", result.stdout)

    if result.returncode != 0:
        print(output)
        raise RuntimeError(f"make failed with exit code {result.returncode}")
    if re.search(r"Verification:\s*Failed", output):
        print(output)
        raise RuntimeError("verification failed; benchmark data was not saved")

    times = {}
    case = None
    dtype = None
    for line in output.splitlines():
        if match := CASE_RE.search(line):
            case = int(match.group(1))
            operator_name = (
                "rms_norm" if match.group(2).lower() == "rmsnorm" else "attention"
            )
            dtype = None
        elif case is not None and (match := DTYPE_RE.search(line)):
            dtype = match.group(1).lower()
        elif case is not None and dtype and (match := TIME_RE.search(line)):
            times[f"{operator_name}_case{case}_{dtype}"] = float(match.group(1))

    if not times:
        print(output)
        raise RuntimeError("no RMS Norm or Attention Avg Time entries found")
    return times


def key_order(key):
    match = re.fullmatch(r"(rms_norm|attention)_case(\d+)_(\w+)", key)
    operator = match.group(1) if match else key
    case = int(match.group(2)) if match else 0
    dtype = match.group(3) if match else key
    return 0 if operator == "rms_norm" else 1, case, 0 if dtype == "float" else 1


def quartiles(values):
    if len(values) == 1:
        return values[0], values[0]
    values = sorted(values)
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    return cuts[0], cuts[2]


def summarize(values):
    q1, q3 = quartiles(values)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    if len(values) >= 4:
        filtered = [
            value for value in values if lower_fence <= value <= upper_fence
        ]
        outliers = [
            value for value in values if value < lower_fence or value > upper_fence
        ]
    else:
        filtered = list(values)
        outliers = []

    p25, p75 = quartiles(filtered)
    return {
        "samples_ms": values,
        "filtered_samples_ms": filtered,
        "outliers_ms": outliers,
        "tukey_lower_fence_ms": lower_fence,
        "tukey_upper_fence_ms": upper_fence,
        "mean_ms": statistics.fmean(filtered),
        "median_ms": statistics.median(filtered),
        "p25_ms": p25,
        "p75_ms": p75,
        "min_ms": min(filtered),
        "max_ms": max(filtered),
    }


def collect(runs, operator):
    samples = {}
    expected_keys = None

    for run in range(1, runs + 1):
        print(f"[{run}/{runs}] running {operator}: make VERBOSE=true ...")
        times = run_make(operator)
        keys = set(times)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise RuntimeError("the set of parsed test cases changed between runs")

        for key, value in times.items():
            samples.setdefault(key, []).append(value)

    results = {}
    for key in sorted(samples, key=key_order):
        results[key] = summarize(samples[key])
    return results


def center(value):
    return value.get("median_ms", value["mean_ms"])


def print_results(results, baseline=None):
    print("\nResult (ms)")
    print(
        f"{'case':<30} {'median':>12} {'p25':>12} {'p75':>12} "
        f"{'outliers':>9} {'speedup':>10}"
    )
    for key, value in results.items():
        speedup = "-"
        if baseline and key in baseline:
            speedup = f"{center(baseline[key]) / center(value):.3f}x"
        print(
            f"{key:<30} {center(value):>12.6f} "
            f"{value['p25_ms']:>12.6f} {value['p75_ms']:>12.6f} "
            f"{len(value['outliers_ms']):>9} {speedup:>10}"
        )


def save_json(path, runs, operator, results):
    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "make VERBOSE=true",
        "operator": operator,
        "runs": runs,
        "outlier_rule": "Tukey fences: Q1 - 1.5*IQR, Q3 + 1.5*IQR",
        "results": results,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"baseline file not found: {path}; run once with --baseline first"
        )
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def plot_results(path, current, baseline=None):
    import matplotlib.pyplot as plt

    suffix = path.suffix or ".png"
    stem = path.stem if path.suffix else path.name
    output_paths = []

    for dtype in ("float", "half"):
        keys = [key for key in current if key.endswith(f"_{dtype}")]
        if not keys:
            continue

        labels = [
            key.removesuffix(f"_{dtype}")
            .replace("rms_norm_", "rms_norm\n")
            .replace("attention_", "attention\n")
            for key in keys
        ]
        x = list(range(len(keys)))
        baseline_keys = [key for key in keys if baseline and key in baseline]
        baseline_indices = [keys.index(key) for key in baseline_keys]
        width = 0.38 if baseline_keys else 0.65
        fig, ax = plt.subplots(figsize=(max(12, len(keys) * 0.55), 7))

        def values_and_errors(results):
            centers = [center(results[key]) for key in keys]
            lower = [
                center(results[key])
                - results[key].get("p25_ms", results[key]["min_ms"])
                for key in keys
            ]
            upper = [
                results[key].get("p75_ms", results[key]["max_ms"])
                - center(results[key])
                for key in keys
            ]
            return centers, [lower, upper]

        current_centers, current_errors = values_and_errors(current)
        if baseline_keys:
            original_keys = keys
            keys = baseline_keys
            baseline_centers, baseline_errors = values_and_errors(baseline)
            keys = original_keys
            ax.bar(
                [x[index] - width / 2 for index in baseline_indices],
                baseline_centers,
                width,
                yerr=baseline_errors,
                capsize=2,
                label="Baseline",
                color="#9aa0a6",
            )
            matched = set(baseline_indices)
            current_x = [
                value + width / 2 if index in matched else value
                for index, value in enumerate(x)
            ]
        else:
            current_x = x

        ax.bar(
            current_x,
            current_centers,
            width,
            yerr=current_errors,
            capsize=2,
            label="Current",
            color="#4285f4",
        )
        ax.set_title(f"RMS Norm and Flash Attention benchmark ({dtype})")
        ax.set_ylabel("Median time (ms), error bars: P25-P75")
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()

        output_path = path.with_name(f"{stem}_{dtype}{suffix}")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RMS Norm and Flash Attention cases"
    )
    parser.add_argument("--runs", type=int, default=5, help="number of make runs")
    parser.add_argument(
        "--operator",
        choices=["all", "rms_norm", "attention"],
        default="all",
        help="operator to benchmark",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="save this run as the baseline"
    )
    parser.add_argument("--plot", action="store_true", help="save a benchmark chart")
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("benchmark_baseline.json"),
    )
    parser.add_argument(
        "--plot-file",
        type=Path,
        default=Path("benchmark_comparison.png"),
        help="output filename prefix; _float and _half are appended",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    results = collect(args.runs, args.operator)
    baseline = None
    if args.baseline:
        save_json(args.baseline_file, args.runs, args.operator, results)
        print(f"\nBaseline saved to {args.baseline_file}")
    else:
        baseline = load_json(args.baseline_file)

    print_results(results, baseline)

    if args.plot:
        plot_paths = plot_results(
            args.plot_file, results, None if args.baseline else baseline
        )
        for plot_path in plot_paths:
            print(f"\nChart saved to {plot_path}")


if __name__ == "__main__":
    main()
