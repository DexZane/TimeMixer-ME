import argparse
import csv
import re
from pathlib import Path

import numpy as np


MODES = ["baseline", "random", "block", "mixed"]
METRIC_NAMES = ["mae", "mse", "rmse", "mape", "mspe"]


def parse_setting(setting: str):
    pred_len_match = re.search(r"_pl(\d+)_", setting)
    seq_len_match = re.search(r"_sl(\d+)_", setting)
    mode_match = re.search(r"(baseline|random|block|mixed)", setting)
    is_ablation = "robust_ablation" in setting.lower() or "robustablation" in setting.lower()

    pred_len = int(pred_len_match.group(1)) if pred_len_match else None
    seq_len = int(seq_len_match.group(1)) if seq_len_match else None
    mode = mode_match.group(1) if mode_match else None

    return {
        "setting": setting,
        "pred_len": pred_len,
        "seq_len": seq_len,
        "mode": mode,
        "is_ablation": is_ablation,
    }


def load_rows(results_root: Path, require_ablation: bool):
    rows = []
    for metrics_path in results_root.glob("*/metrics.npy"):
        setting = metrics_path.parent.name
        parsed = parse_setting(setting)
        if require_ablation and not parsed["is_ablation"]:
            continue
        if parsed["mode"] not in MODES:
            continue
        if parsed["pred_len"] is None:
            continue

        metrics = np.load(metrics_path)
        if metrics.shape[0] < 5:
            continue

        row = {
            "setting": setting,
            "metrics_path": str(metrics_path),
            "seq_len": parsed["seq_len"],
            "pred_len": parsed["pred_len"],
            "mode": parsed["mode"],
            "mae": float(metrics[0]),
            "mse": float(metrics[1]),
            "rmse": float(metrics[2]),
            "mape": float(metrics[3]),
            "mspe": float(metrics[4]),
        }
        rows.append(row)

    rows.sort(key=lambda x: (x["pred_len"], MODES.index(x["mode"]), x["setting"]))
    return rows


def write_raw_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["setting", "metrics_path", "seq_len", "pred_len", "mode"] + METRIC_NAMES
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_metric_table(rows, metric_name: str):
    pred_lens = sorted({r["pred_len"] for r in rows})
    table = {}
    for pred_len in pred_lens:
        table[pred_len] = {m: None for m in MODES}
    for r in rows:
        table[r["pred_len"]][r["mode"]] = r[metric_name]
    return pred_lens, table


def write_metric_csv(rows, metric_name: str, output_path: Path):
    pred_lens, table = build_metric_table(rows, metric_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pred_len"] + MODES)
        for pred_len in pred_lens:
            values = [table[pred_len][m] for m in MODES]
            writer.writerow([pred_len] + values)


def write_markdown(rows, output_path: Path):
    pred_lens = sorted({r["pred_len"] for r in rows})
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Robustness Ablation Summary")
    lines.append("")

    for metric_name in ["mse", "mae", "rmse"]:
        _, table = build_metric_table(rows, metric_name)
        lines.append(f"## {metric_name.upper()}")
        lines.append("")
        lines.append("| pred_len | baseline | random | block | mixed |")
        lines.append("|---:|---:|---:|---:|---:|")
        for pred_len in pred_lens:
            values = []
            for m in MODES:
                val = table[pred_len][m]
                values.append("NA" if val is None else f"{val:.6f}")
            lines.append(f"| {pred_len} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize robustness ablation results from metrics.npy files.")
    parser.add_argument("--results_root", type=str, default="./results", help="root directory containing setting folders")
    parser.add_argument("--output_dir", type=str, default="./results/ablation_summary", help="output directory")
    parser.add_argument("--include_non_ablation", action="store_true",
                        help="include folders not containing 'robust_ablation' in setting name")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    require_ablation = not args.include_non_ablation

    rows = load_rows(results_root=results_root, require_ablation=require_ablation)
    if not rows:
        print("No robustness ablation results found.")
        print(f"Searched: {results_root.resolve()}")
        return

    raw_csv = output_dir / "robustness_ablation_raw.csv"
    mse_csv = output_dir / "robustness_ablation_mse.csv"
    mae_csv = output_dir / "robustness_ablation_mae.csv"
    rmse_csv = output_dir / "robustness_ablation_rmse.csv"
    md_path = output_dir / "robustness_ablation_summary.md"

    write_raw_csv(rows, raw_csv)
    write_metric_csv(rows, "mse", mse_csv)
    write_metric_csv(rows, "mae", mae_csv)
    write_metric_csv(rows, "rmse", rmse_csv)
    write_markdown(rows, md_path)

    print(f"Rows: {len(rows)}")
    print(f"Saved: {raw_csv.resolve()}")
    print(f"Saved: {mse_csv.resolve()}")
    print(f"Saved: {mae_csv.resolve()}")
    print(f"Saved: {rmse_csv.resolve()}")
    print(f"Saved: {md_path.resolve()}")


if __name__ == "__main__":
    main()
