import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _as_sequence_list(values) -> List[np.ndarray]:
    if isinstance(values, np.ndarray) and values.dtype != object:
        if values.ndim == 2:
            values = values[..., None]
        return [np.asarray(sample, dtype=np.float32) for sample in values]

    if isinstance(values, np.ndarray) and values.dtype == object:
        iterable = values.tolist()
    else:
        iterable = list(values)

    sequences = []
    for sample in iterable:
        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim == 1:
            sample = sample[:, None]
        sequences.append(sample)
    return sequences


def _prepare_labels(labels) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    return labels


def _build_label_mapping(*label_arrays):
    flattened = []
    for labels in label_arrays:
        flattened.extend(_prepare_labels(labels).tolist())

    if not flattened:
        raise ValueError("Empty classification label set")

    if all(np.isscalar(label) and isinstance(label, (int, float, np.integer, np.floating)) for label in flattened):
        unique_labels = sorted(set(flattened))
    else:
        unique_labels = sorted(set(flattened), key=lambda item: str(item))

    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    class_names = [str(label) for label in unique_labels]
    return mapping, class_names


def _encode_labels(labels, mapping) -> np.ndarray:
    labels = _prepare_labels(labels)
    return np.asarray([mapping[label] for label in labels.tolist()], dtype=np.int64)


def _load_npz(root_path: str):
    for file_name in ("classification.npz", "dataset.npz", "data.npz"):
        path = os.path.join(root_path, file_name)
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True)
        keys = set(data.keys())
        key_groups = [
            ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test"),
            ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y"),
        ]
        for key_group in key_groups:
            if key_group[0] in keys and key_group[1] in keys and key_group[4] in keys and key_group[5] in keys:
                return {
                    "train": (_as_sequence_list(data[key_group[0]]), _prepare_labels(data[key_group[1]])),
                    "val": (
                        _as_sequence_list(data[key_group[2]]),
                        _prepare_labels(data[key_group[3]]),
                    ) if key_group[2] in keys and key_group[3] in keys else None,
                    "test": (_as_sequence_list(data[key_group[4]]), _prepare_labels(data[key_group[5]])),
                }
        raise ValueError(f"Unsupported classification npz format in {path}")
    return None


def _load_npy_pairs(root_path: str):
    split_aliases = {
        "train": [("train.npy", "train_labels.npy"), ("x_train.npy", "y_train.npy")],
        "val": [("val.npy", "val_labels.npy"), ("x_val.npy", "y_val.npy")],
        "test": [("test.npy", "test_labels.npy"), ("x_test.npy", "y_test.npy")],
    }
    loaded = {}
    for split, candidates in split_aliases.items():
        for data_name, label_name in candidates:
            data_path = os.path.join(root_path, data_name)
            label_path = os.path.join(root_path, label_name)
            if os.path.exists(data_path) and os.path.exists(label_path):
                data = np.load(data_path, allow_pickle=True)
                labels = np.load(label_path, allow_pickle=True)
                loaded[split] = (_as_sequence_list(data), _prepare_labels(labels))
                break
    if "train" in loaded and "test" in loaded:
        loaded.setdefault("val", None)
        return loaded
    return None


def _split_train_val(
    train_x: Sequence[np.ndarray],
    train_y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[Tuple[List[np.ndarray], np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_y))
    val_indices = []
    for cls in np.unique(train_y):
        cls_idx = indices[train_y == cls]
        rng.shuffle(cls_idx)
        if len(cls_idx) <= 1:
            continue
        cls_val_count = max(1, int(round(len(cls_idx) * val_ratio)))
        if cls_val_count >= len(cls_idx):
            cls_val_count = max(1, len(cls_idx) - 1)
        val_indices.append(cls_idx[:cls_val_count])
    val_indices = np.concatenate(val_indices) if val_indices else np.array([], dtype=np.int64)
    if len(val_indices) == 0 and len(indices) > 1:
        rng.shuffle(indices)
        val_indices = indices[:1]
    train_mask = np.ones(len(train_y), dtype=bool)
    train_mask[val_indices] = False
    new_train_x = [train_x[idx] for idx in indices[train_mask]]
    new_train_y = train_y[train_mask]
    val_x = [train_x[idx] for idx in val_indices]
    val_y = train_y[val_indices]
    return (new_train_x, new_train_y), (val_x, val_y)


def load_classification_splits(root_path: str, val_ratio: float = 0.2, seed: int = 42):
    loaded = _load_npz(root_path)
    if loaded is None:
        loaded = _load_npy_pairs(root_path)
    if loaded is None:
        raise FileNotFoundError(
            "Classification data not found. Expected classification.npz/dataset.npz "
            "or train/test .npy split files under root_path."
        )

    train_x, train_y = loaded["train"]
    test_x, test_y = loaded["test"]
    val_split = loaded.get("val")
    if val_split is None:
        (train_x, train_y), val_split = _split_train_val(train_x, train_y, val_ratio=val_ratio, seed=seed)

    val_x, val_y = val_split
    label_mapping, class_names = _build_label_mapping(train_y, val_y, test_y)
    return {
        "train": (train_x, _encode_labels(train_y, label_mapping)),
        "val": (val_x, _encode_labels(val_y, label_mapping)),
        "test": (test_x, _encode_labels(test_y, label_mapping)),
        "class_names": class_names,
    }


class ClassificationDataset(Dataset):
    def __init__(self, root_path, flag="train", val_ratio=0.2, seed=42):
        assert flag in ["train", "val", "test"]
        splits = load_classification_splits(root_path, val_ratio=val_ratio, seed=seed)
        self.samples, self.labels = splits[flag]

        all_labels = np.concatenate([splits["train"][1], splits["val"][1], splits["test"][1]])
        num_features = self.samples[0].shape[-1]
        self.max_seq_len = max(sample.shape[0] for sample in self.samples)
        self.global_max_seq_len = max(
            max(sample.shape[0] for sample in splits["train"][0]),
            max(sample.shape[0] for sample in splits["val"][0]),
            max(sample.shape[0] for sample in splits["test"][0]),
        )
        self.feature_df = np.zeros((1, num_features), dtype=np.float32)
        self.class_names = splits["class_names"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], int(self.labels[idx])


def classification_collate_fn(data):
    batch_x, labels = zip(*data)
    max_len = max(item.shape[0] for item in batch_x)
    feature_dim = batch_x[0].shape[-1]

    padded = torch.zeros(len(batch_x), max_len, feature_dim, dtype=torch.float32)
    padding_masks = torch.zeros(len(batch_x), max_len, dtype=torch.float32)

    for i, sample in enumerate(batch_x):
        seq_len = min(sample.shape[0], max_len)
        padded[i, :seq_len] = torch.as_tensor(sample[:seq_len], dtype=torch.float32)
        padding_masks[i, :seq_len] = 1.0

    labels = torch.as_tensor(labels, dtype=torch.long)
    return padded, labels, padding_masks
