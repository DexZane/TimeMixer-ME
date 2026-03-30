import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from models import TimeMixer_ME


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


def save_heatmap(matrix, labels, title, output_path, max_labels=30):
    n = matrix.shape[0]
    if n > max_labels:
        labels = [str(i) for i in range(n)]

    fig_size = max(6, min(16, n * 0.35))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xticks(range(n), labels, rotation=90, fontsize=8)
    plt.yticks(range(n), labels, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def reduce_attn(attn_tensor):
    # channel_independence=1: [B, T, N, N] -> [N, N]
    # channel_independence=0: [B, T, T] -> [T, T]
    if attn_tensor is None:
        return None
    arr = attn_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        return arr.mean(axis=(0, 1))
    if arr.ndim == 3:
        return arr.mean(axis=0)
    raise ValueError(f"Unexpected attention tensor shape: {arr.shape}")


def run_attention_explain(args):
    device = torch.device(
        f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    args.output_attention = True
    model = TimeMixer_ME.Model(args).float().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    _, test_loader = data_provider(args, "test")
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    if args.data in ["PEMS", "Solar"]:
        batch_x_mark = None
        batch_y_mark = None

    if args.down_sampling_layers == 0:
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float().to(device)
        dec_inp = torch.cat(
            [batch_y[:, : args.label_len, :].float().to(device), dec_inp], dim=1
        )
    else:
        dec_inp = None

    with torch.no_grad():
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise RuntimeError(
            "Model did not return attention weights. Ensure output_attention is enabled."
        )

    _, all_attn_weights = outputs
    os.makedirs(args.output_dir, exist_ok=True)

    global_mats = []
    layer_stats = []

    for layer_idx, layer_attn_list in enumerate(all_attn_weights):
        if not layer_attn_list:
            continue

        scale_mats = []
        for scale_idx, attn in enumerate(layer_attn_list):
            mat = reduce_attn(attn)
            if mat is None:
                continue
            scale_mats.append(mat)

            labels = infer_feature_names(args, mat.shape[0])
            np.save(
                os.path.join(
                    args.output_dir, f"attn_layer{layer_idx}_scale{scale_idx}.npy"
                ),
                mat,
            )
            save_heatmap(
                mat,
                labels,
                f"Attention Layer {layer_idx} Scale {scale_idx}",
                os.path.join(
                    args.output_dir, f"attn_layer{layer_idx}_scale{scale_idx}.png"
                ),
                max_labels=args.max_labels,
            )

        if not scale_mats:
            continue
        layer_mat = np.mean(np.stack(scale_mats, axis=0), axis=0)
        global_mats.append(layer_mat)

        labels = infer_feature_names(args, layer_mat.shape[0])
        np.save(os.path.join(args.output_dir, f"attn_layer{layer_idx}_mean.npy"), layer_mat)
        save_heatmap(
            layer_mat,
            labels,
            f"Attention Layer {layer_idx} Mean",
            os.path.join(args.output_dir, f"attn_layer{layer_idx}_mean.png"),
            max_labels=args.max_labels,
        )

        layer_stats.append(
            {
                "layer": layer_idx,
                "mean_attn": float(layer_mat.mean()),
                "std_attn": float(layer_mat.std()),
                "diag_mean": float(np.diag(layer_mat).mean()),
            }
        )

    if not global_mats:
        raise RuntimeError("No attention matrices were produced from this batch.")

    global_attn = np.mean(np.stack(global_mats, axis=0), axis=0)
    labels = infer_feature_names(args, global_attn.shape[0])
    np.save(os.path.join(args.output_dir, "attn_global_mean.npy"), global_attn)
    save_heatmap(
        global_attn,
        labels,
        "Global Mean Attention (Across Layers & Scales)",
        os.path.join(args.output_dir, "attn_global_mean.png"),
        max_labels=args.max_labels,
    )

    top_pairs = []
    if global_attn.ndim == 2:
        tmp = global_attn.copy()
        np.fill_diagonal(tmp, -np.inf)
        flat_idx = np.argpartition(tmp.ravel(), -args.top_pairs)[-args.top_pairs :]
        sorted_idx = flat_idx[np.argsort(tmp.ravel()[flat_idx])[::-1]]
        for idx in sorted_idx:
            i, j = np.unravel_index(idx, tmp.shape)
            top_pairs.append(
                {
                    "from_feature": labels[i] if i < len(labels) else str(i),
                    "to_feature": labels[j] if j < len(labels) else str(j),
                    "attention": float(global_attn[i, j]),
                }
            )

    meta = {
        "checkpoint": args.checkpoint,
        "num_layers": len(all_attn_weights),
        "channel_independence": int(args.channel_independence),
        "output_dir": args.output_dir,
        "layer_stats": layer_stats,
        "top_pairs": top_pairs,
    }
    with open(os.path.join(args.output_dir, "attention_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved attention explainability outputs to: {args.output_dir}")
    if top_pairs:
        print("Top attention feature pairs:")
        for item in top_pairs[:10]:
            print(
                f"  {item['from_feature']} -> {item['to_feature']}: {item['attention']:.6f}"
            )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Attention explainability for TimeMixer-ME long_term_forecast"
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/attention_explain")
    parser.add_argument("--top_pairs", type=int, default=20)
    parser.add_argument("--max_labels", type=int, default=30)

    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--model", type=str, default="TimeMixer_ME")
    parser.add_argument("--model_id", type=str, default="attention_explain")

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
    parser.add_argument("--output_attention", action="store_true", default=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--inverse", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    run_attention_explain(cli_args)
