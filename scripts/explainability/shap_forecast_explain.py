import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from models import TimeMixer_ME


class ForecastScalarWrapper(nn.Module):
    def __init__(self, model, pred_index, feature_index):
        super().__init__()
        self.model = model
        self.pred_index = pred_index
        self.feature_index = feature_index

    def forward(self, x_enc):
        outputs = self.model(x_enc, None, None, None)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        scalar = outputs[:, self.pred_index, self.feature_index]
        return scalar.unsqueeze(-1)


def infer_feature_names(args, num_features):
    csv_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(csv_path):
        return [f"feature_{i}" for i in range(num_features)]

    try:
        df = pd.read_csv(csv_path, nrows=1)
    except Exception:
        return [f"feature_{i}" for i in range(num_features)]

    if args.features in ["M", "MS"]:
        cols = list(df.columns[1:])
    else:
        cols = [args.target]

    if len(cols) == num_features:
        return cols
    return [f"feature_{i}" for i in range(num_features)]


def select_samples(batch_x, background_size, explain_size):
    total = batch_x.size(0)
    if total < 2:
        raise ValueError("Batch size must be >= 2 for SHAP analysis.")

    background_size = min(background_size, total - 1)
    explain_size = min(explain_size, total - background_size)
    if explain_size <= 0:
        explain_size = 1

    background = batch_x[:background_size]
    explain = batch_x[background_size : background_size + explain_size]
    return background, explain


def run_shap(args):
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    device = torch.device(
        f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    model = TimeMixer_ME.Model(args).float().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    test_set, test_loader = data_provider(args, "test")
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
    batch_x = batch_x.float().to(device)

    background_x, explain_x = select_samples(
        batch_x, args.background_size, args.explain_size
    )
    pred_index = min(max(args.pred_index, 0), args.pred_len - 1)
    feature_index = min(max(args.feature_index, 0), args.c_out - 1)

    wrapped = ForecastScalarWrapper(model, pred_index, feature_index).to(device)
    explainer = shap.GradientExplainer(wrapped, background_x)
    shap_values = explainer.shap_values(explain_x)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    # [B, T, N] -> [N] channel-level attribution score
    channel_importance = np.mean(np.abs(shap_values), axis=(0, 1))
    feature_names = infer_feature_names(args, channel_importance.shape[0])

    result_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": channel_importance}
    ).sort_values("mean_abs_shap", ascending=False)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "shap_values.npy"), shap_values)
    result_df.to_csv(
        os.path.join(args.output_dir, "feature_importance.csv"), index=False
    )

    top_k = min(args.plot_top_k, len(result_df))
    top_df = result_df.head(top_k).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"], top_df["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Input feature")
    plt.title(
        f"TimeMixer-ME SHAP Importance (pred_t={pred_index}, out_feature={feature_index})"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "feature_importance_topk.png"), dpi=200)
    plt.close()

    meta = {
        "checkpoint": args.checkpoint,
        "pred_index": pred_index,
        "feature_index": feature_index,
        "background_size": int(background_x.size(0)),
        "explain_size": int(explain_x.size(0)),
        "output_dir": args.output_dir,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved SHAP results to: {args.output_dir}")
    print(result_df.head(10).to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser(description="SHAP explanation for TimeMixer-ME")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/shap_explain")
    parser.add_argument("--background_size", type=int, default=16)
    parser.add_argument("--explain_size", type=int, default=16)
    parser.add_argument("--plot_top_k", type=int, default=15)
    parser.add_argument("--pred_index", type=int, default=0)
    parser.add_argument("--feature_index", type=int, default=0)

    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--model", type=str, default="TimeMixer_ME")
    parser.add_argument("--model_id", type=str, default="shap_explain")

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)

    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--decomp_method", type=str, default="moving_avg")
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=0)
    parser.add_argument("--down_sampling_window", type=int, default=1)
    parser.add_argument("--down_sampling_method", type=str, default="avg")
    parser.add_argument("--use_future_temporal_feature", type=int, default=0)

    parser.add_argument("--num_memories", type=int, default=32)
    parser.add_argument("--causal_levels", type=int, default=4)
    parser.add_argument("--output_attention", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    run_shap(cli_args)
