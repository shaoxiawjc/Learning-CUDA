"""Compare benchmark results across named runs.

Usage:
    .venv/bin/python compare.py --runs v1,v2 --dtype fp16 --rows 1024,4096 --dims 512
    .venv/bin/python compare.py --metric ours_gflops --dtype bf16
"""
import argparse
import json
import os

METRICS = {
    "speedup": "speedup",
    "ours_ms": "ours_ms",
    "fht_ms": "fht_ms",
    "hada_ms": "hada_ms",
    "ours_gflops": "ours_gflops",
    "fht_gflops": "fht_gflops",
    "hada_gflops": "hada_gflops",
    "ours_gbps": "ours_gbps",
    "fht_gbps": "fht_gbps",
    "hada_gbps": "hada_gbps",
    "speedup_hada": "speedup_hada",
}

DTYPES = {"fp16": "float16", "bf16": "bfloat16", "all": None}


def parse_ints(spec):
    if spec is None or spec == "all":
        return None
    return {int(t) for t in spec.split(",")}


def fmt(v, metric):
    if v is None:
        return "-" * 9
    if metric.startswith("speedup"):
        return f"{v:8.2f}x"
    if metric.endswith("ms"):
        return f"{v:9.3f}"
    return f"{v:9.1f}"


def main():
    parser = argparse.ArgumentParser(description="Compare saved bench results across runs.")
    parser.add_argument("--file", default="bench_results.json", help="results JSON to read")
    parser.add_argument("--runs", default="all", help="comma list of run names, or 'all'")
    parser.add_argument("--dtype", default="all", choices=list(DTYPES), help="which dtype to show")
    parser.add_argument("--rows", default="all", help="comma list of row counts, or 'all'")
    parser.add_argument("--dims", default="all", help="comma list of dims, or 'all'")
    parser.add_argument("--metric", default="speedup", choices=list(METRICS),
                        help="which metric to compare")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"no results file '{args.file}'")

    with open(args.file) as f:
        data = json.load(f)

    if args.runs == "all":
        runs = list(data.keys())
    else:
        runs = [r.strip() for r in args.runs.split(",")]
    missing = [r for r in runs if r not in data]
    if missing:
        raise SystemExit(f"run(s) {missing} not found; available: {list(data.keys())}")

    dt = DTYPES[args.dtype]
    rows_f = parse_ints(args.rows)
    dims_f = parse_ints(args.dims)
    metric = METRICS[args.metric]

    print(f"metric: {args.metric}")
    for rn in runs:
        meta = data[rn]
        print(f"  [{rn}] {meta.get('gpu', '?')} @ {meta.get('timestamp', '?')}")
    print()

    dtypes = ("float16", "bfloat16") if dt is None else (dt,)
    cell_w = 9
    col_w = max(cell_w, max(len(rn) for rn in runs))

    for d in dtypes:
        # shapes (rows, cols) present across selected runs under this dtype + filters
        shapes = set()
        for rn in runs:
            for res in data[rn]["results"]:
                if res["dtype"] != d:
                    continue
                if rows_f is not None and res["rows"] not in rows_f:
                    continue
                if dims_f is not None and res["cols"] not in dims_f:
                    continue
                shapes.add((res["rows"], res["cols"]))
        if not shapes:
            continue
        shapes = sorted(shapes)

        # index: run -> (rows, cols) -> result
        idx = {}
        for rn in runs:
            m = {}
            for res in data[rn]["results"]:
                if res["dtype"] == d:
                    m[(res["rows"], res["cols"])] = res
            idx[rn] = m

        shape_w = max(len(f"{r}x{c}") for r, c in shapes)
        print(f"dtype={d}")
        header = f"{'shape':<{shape_w}}  " + "  ".join(f"{rn:>{col_w}}" for rn in runs)
        print(header)
        print("-" * len(header))
        for r, c in shapes:
            cells = []
            for rn in runs:
                res = idx[rn].get((r, c))
                cells.append(fmt(res[metric], args.metric) if res else "-" * cell_w)
            print(f"{f'{r}x{c}':<{shape_w}}  " + "  ".join(cells))
        print()


if __name__ == "__main__":
    main()
