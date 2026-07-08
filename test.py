# -*- coding: utf-8 -*-
"""
test_ccscd5_dataset_diffreg_trustH_3class.py

三分类 BCD + multi 模式语义分割测试脚本，适配：
- model_3_6_stabilized_differentiable_registration_trustH.py
- BCD 标签：0 = unchanged, 1 = changed, 2 = unknown / non-overlap
- 可选真实 H0_t1_to_t2：推荐与训练时保持一致

Expected directory layout:
    data_root/
        bcd/
        curr_img/
        curr_seg/
        prev_img/
        prev_seg/
        homography/          # optional，每个样本一个 .npy/.txt/.csv 3x3 矩阵
        crop_meta.csv        # optional，可用 start_x/start_y/start_x2/start_y2 或 shift_x/shift_y 生成 H

Model forward convention:
    T1 = prev_img
    T2 = curr_img
    Outputs are in T2 coordinates by default.

推荐测试命令：
    python test_ccscd5_dataset_diffreg_trustH_3class.py ^
      --model_module model_3_6_stabilized_differentiable_registration_trustH ^
      --checkpoint weights_save/stabilized_diffreg/best.pth ^
      --data_root ./datasets_made/SCSCD7/test ^
      --bcd_num_classes 3 ^
      --bcd_label_mode already_012_255 ^
      --use_provided_homography ^
      --homography_csv crop_meta.csv ^
      --save_pred ^
      --save_seg ^
      --save_scd ^
      --save_inputs
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Basic constants
# =============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

CC_SCD5_CLASSES = [
    "Others",
    "Water",
    "Building",
    "Vegetation",
    "Road",
]

SC_SCD7_CLASSES = [
    "Bareland",
    "Water",
    "Building",
    "Structure",
    "Farmland",
    "Vegetation",
    "Road",
]

# RGB palettes for semantic visualization. These are fixed, not random.
# The same semantic category uses the same color across CC-SCD5 and SC-SCD7 when categories overlap.
CC_SCD5_PALETTE = np.array([
    [0, 0, 0],        # 0 Others - black; CC-SCD5 specific
    [0, 0, 255],      # 1 Water
    [0, 255, 0],      # 2 Building
    [0, 128, 0],      # 3 Vegetation
    [128, 64, 0],     # 4 Road
], dtype=np.uint8)

SC_SCD7_PALETTE = np.array([
    [128, 128, 128],  # 0 Bareland
    [0, 0, 255],      # 1 Water
    [0, 255, 0],      # 2 Building
    [255, 0, 0],      # 3 Structure
    [0, 255, 255],    # 4 Farmland
    [0, 128, 0],      # 5 Vegetation
    [128, 64, 0],     # 6 Road
], dtype=np.uint8)

BCD_CLASS_NAMES = {
    0: "unchanged",
    1: "changed",
    2: "unknown_nonoverlap",
}


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Dataset utilities
# =============================================================================

def _list_image_files(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Required folder not found: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def _index_by_stem(files: Sequence[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in files:
        out.setdefault(p.stem, p)
    return out


def _read_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _read_bcd(path: Path, mode: str = "already_012_255") -> torch.Tensor:
    """
    Returns long target [H,W].

    Supported conventions:
    - binary_0_255: all non-zero pixels are changed=1.
    - already_012_255: keep labels 0/1/2/255; invalid labels become 255.
    - rgb: black=0, red/white=1, blue=2, everything else=255.
    """
    if mode == "rgb":
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img)
        target = np.full(arr.shape[:2], 255, dtype=np.uint8)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        black = (r < 20) & (g < 20) & (b < 20)
        red = (r > 128) & (g < 80) & (b < 80)
        white = (r > 128) & (g > 128) & (b > 128)
        blue = (b > 128) & (r < 80) & (g < 120)
        target[black] = 0
        target[red | white] = 1
        target[blue] = 2
    else:
        img = Image.open(path).convert("L")
        arr = np.asarray(img)
        if mode == "binary_0_255":
            target = (arr > 0).astype(np.uint8)
        elif mode == "already_012_255":
            target = arr.astype(np.uint8)
            valid = np.isin(target, np.array([0, 1, 2, 255], dtype=np.uint8))
            target[~valid] = 255
        else:
            raise ValueError(f"Unsupported BCD label mode: {mode}")
    return torch.from_numpy(target).long().contiguous()


def _read_seg(path: Path, mode: str = "grayscale", ignore_value: int = 255, num_classes: int = 7) -> torch.Tensor:
    if mode != "grayscale":
        raise ValueError(f"Unsupported segmentation label mode: {mode}")
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.int64)
    if ignore_value >= 0:
        arr[arr == ignore_value] = 255
    valid = ((arr >= 0) & (arr < num_classes)) | (arr == 255)
    arr[~valid] = 255
    return torch.from_numpy(arr).long().contiguous()


def _load_homography_matrix(path: Path) -> np.ndarray:
    """
    Load a 3x3 homography from .npy/.txt/.csv.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        H = np.load(path)
    elif suffix in {".txt", ".csv"}:
        H = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"Unsupported homography file type: {path}")
    H = np.asarray(H, dtype=np.float32)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be shape [3,3], got {H.shape} from {path}")
    if abs(float(H[2, 2])) < 1e-8:
        raise ValueError(f"Invalid homography H[2,2] is zero: {path}")
    H = H / H[2, 2]
    return H.astype(np.float32)


def _translation_homography_from_offsets(shift_x: float, shift_y: float) -> np.ndarray:
    """
    T1 crop -> T2 crop.
    If T2 crop origin = T1 crop origin + (shift_x, shift_y),
    then x_T2 = x_T1 - shift_x, y_T2 = y_T1 - shift_y.
    """
    return np.array(
        [[1.0, 0.0, -float(shift_x)],
         [0.0, 1.0, -float(shift_y)],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _load_crop_meta_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Load sample-wise homographies from crop_meta.csv.

    Supported columns:
    1) name, shift_x, shift_y
    2) name, start_x, start_y, start_x2, start_y2
       shift_x = start_x2 - start_x
       shift_y = start_y2 - start_y

    The "name" may include extension; both stem and full name are indexed.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Homography CSV not found: {csv_path}")

    out: Dict[str, np.ndarray] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")

        fieldnames = set(reader.fieldnames)
        has_shift = {"name", "shift_x", "shift_y"}.issubset(fieldnames)
        has_starts = {"name", "start_x", "start_y", "start_x2", "start_y2"}.issubset(fieldnames)
        if not has_shift and not has_starts:
            raise ValueError(
                f"{csv_path} must contain either columns "
                "name,shift_x,shift_y or name,start_x,start_y,start_x2,start_y2. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            raw_name = str(row["name"]).strip()
            if not raw_name:
                continue
            if has_shift:
                shift_x = float(row["shift_x"])
                shift_y = float(row["shift_y"])
            else:
                start_x = float(row["start_x"])
                start_y = float(row["start_y"])
                start_x2 = float(row["start_x2"])
                start_y2 = float(row["start_y2"])
                shift_x = start_x2 - start_x
                shift_y = start_y2 - start_y

            H = _translation_homography_from_offsets(shift_x, shift_y)
            out[raw_name] = H
            out[Path(raw_name).stem] = H

    return out


def _find_homography_file(homography_dir: Path, stem: str) -> Optional[Path]:
    for ext in [".npy", ".txt", ".csv"]:
        p = homography_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


class CCSCD5PairTestDataset(Dataset):
    """
    Dataset that pairs files by the same filename stem.
    Optional sample-wise H0_t1_to_t2 can be loaded from:
      - data_root / homography_dir / <stem>.npy|txt|csv
      - data_root / homography_csv
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        bcd_dir: str = "bcd",
        curr_img_dir: str = "curr_img",
        curr_seg_dir: str = "curr_seg",
        prev_img_dir: str = "prev_img",
        prev_seg_dir: str = "prev_seg",
        bcd_label_mode: str = "already_012_255",
        seg_label_mode: str = "grayscale",
        seg_ignore_value: int = 255,
        seg_num_classes: int = 7,
        use_provided_homography: bool = False,
        homography_dir: str = "homography",
        homography_csv: str = "crop_meta.csv",
        allow_missing_homography: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.bcd_label_mode = bcd_label_mode
        self.seg_label_mode = seg_label_mode
        self.seg_ignore_value = seg_ignore_value
        self.seg_num_classes = seg_num_classes
        self.use_provided_homography = bool(use_provided_homography)
        self.homography_dir = homography_dir
        self.homography_csv = homography_csv
        self.allow_missing_homography = bool(allow_missing_homography)

        bcd_files = _list_image_files(self.data_root / bcd_dir)
        curr_img = _index_by_stem(_list_image_files(self.data_root / curr_img_dir))
        curr_seg = _index_by_stem(_list_image_files(self.data_root / curr_seg_dir))
        prev_img = _index_by_stem(_list_image_files(self.data_root / prev_img_dir))
        prev_seg = _index_by_stem(_list_image_files(self.data_root / prev_seg_dir))

        self.csv_homographies: Dict[str, np.ndarray] = {}
        csv_path = self.data_root / self.homography_csv
        if self.use_provided_homography and self.homography_csv and csv_path.exists():
            self.csv_homographies = _load_crop_meta_csv(csv_path)

        samples: List[Dict[str, Any]] = []
        missing: List[str] = []
        missing_h: List[str] = []

        for bcd_path in bcd_files:
            key = bcd_path.stem
            paths: Dict[str, Optional[Path]] = {
                "bcd": bcd_path,
                "curr_img": curr_img.get(key),
                "curr_seg": curr_seg.get(key),
                "prev_img": prev_img.get(key),
                "prev_seg": prev_seg.get(key),
            }
            if not all(v is not None for v in paths.values()):
                missing.append(key)
                continue

            H_file: Optional[Path] = None
            has_H = False
            if self.use_provided_homography:
                if key in self.csv_homographies or bcd_path.name in self.csv_homographies:
                    has_H = True
                else:
                    H_file = _find_homography_file(self.data_root / self.homography_dir, key)
                    has_H = H_file is not None

                if not has_H and not self.allow_missing_homography:
                    missing_h.append(key)
                    continue

            sample: Dict[str, Any] = {k: v for k, v in paths.items() if v is not None}
            sample["H_file"] = H_file
            samples.append(sample)

        if not samples:
            extra = ""
            if missing_h:
                extra = f" Missing homography count={len(missing_h)}, first={missing_h[:10]}"
            raise RuntimeError(
                "No valid samples found. Files must share the same stem across "
                "bcd/curr_img/curr_seg/prev_img/prev_seg."
                + extra
            )

        self.samples = samples
        self.missing = missing
        self.missing_h = missing_h

    def __len__(self) -> int:
        return len(self.samples)

    def _get_homography(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            H: [3,3] float32
            has_H: scalar bool tensor
        """
        if not self.use_provided_homography:
            return torch.eye(3, dtype=torch.float32), torch.tensor(False)

        bcd_path: Path = sample["bcd"]
        key = bcd_path.stem
        if key in self.csv_homographies:
            H = self.csv_homographies[key]
            return torch.from_numpy(H).float(), torch.tensor(True)
        if bcd_path.name in self.csv_homographies:
            H = self.csv_homographies[bcd_path.name]
            return torch.from_numpy(H).float(), torch.tensor(True)

        H_file = sample.get("H_file")
        if H_file is not None:
            H = _load_homography_matrix(Path(H_file))
            return torch.from_numpy(H).float(), torch.tensor(True)

        return torch.eye(3, dtype=torch.float32), torch.tensor(False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        H, has_H = self._get_homography(s)
        return {
            "t1": _read_rgb(s["prev_img"]),
            "t2": _read_rgb(s["curr_img"]),
            "bcd": _read_bcd(s["bcd"], self.bcd_label_mode),
            "label_a": _read_seg(s["prev_seg"], self.seg_label_mode, self.seg_ignore_value, self.seg_num_classes),
            "label_b": _read_seg(s["curr_seg"], self.seg_label_mode, self.seg_ignore_value, self.seg_num_classes),
            "H0_t1_to_t2": H,
            "has_H0": has_H,
            "name": s["bcd"].stem,
        }


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


# =============================================================================
# DINO-only encoder
# =============================================================================

class DINOOnlyEncoder(nn.Module):
    """
    DINO-only feature extractor used by ComprehensiveUnregisteredMultiTaskCD.
    It returns p2,p3,p4,p5 feature maps, each projected to fpn_channels.
    """

    def __init__(
        self,
        fpn_channels: int = 128,
        dino_weight: str = "dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        device: str = "cuda",
        extract_ids: Sequence[int] = (5, 11, 17, 23),
        dino_out_dim: int = 1024,
        freeze_dino: bool = True,
    ) -> None:
        super().__init__()
        from model.blocks.adapter import DINOV3Wrapper, DenseAdapterLite

        self.fpn_channels = int(fpn_channels)
        self.dense_out_dim = self.fpn_channels * 2
        self.dino = DINOV3Wrapper(weights_path=dino_weight, device=device, extract_ids=list(extract_ids))
        self.dense_adp = DenseAdapterLite(
            in_dim=dino_out_dim,
            out_dim=self.dense_out_dim,
            bottleneck=max(self.fpn_channels // 2, 1),
        )
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.dense_out_dim, self.fpn_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, self.fpn_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(4)
        ])
        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False
            self.dino.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if all(not p.requires_grad for p in self.dino.parameters()):
            self.dino.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dino_frozen = all(not p.requires_grad for p in self.dino.parameters())
        if dino_frozen:
            with torch.no_grad():
                ds_fea = self.dino(x)
        else:
            ds_fea = self.dino(x)
        ds_fea = self.dense_adp(ds_fea)
        if not isinstance(ds_fea, (tuple, list)) or len(ds_fea) != 4:
            raise ValueError(
                "DenseAdapterLite must return 4 feature maps. "
                f"Got type={type(ds_fea)}, len={len(ds_fea) if isinstance(ds_fea, (tuple, list)) else 'N/A'}"
            )
        return tuple(proj(f) for proj, f in zip(self.proj, ds_fea))


# =============================================================================
# Model builder / checkpoint loading
# =============================================================================

def _config_kwargs_for_dataclass(config_cls: Any, args: argparse.Namespace) -> Dict[str, Any]:
    if is_dataclass(config_cls):
        names = {f.name for f in fields(config_cls)}
    else:
        names = set(getattr(config_cls, "__annotations__", {}).keys())
    return {name: getattr(args, name) for name in names if hasattr(args, name)}


def build_model(args: argparse.Namespace, device: torch.device):
    module = importlib.import_module(args.model_module)
    config_cls = getattr(module, "ComprehensiveCDConfig")
    model_cls = getattr(module, "ComprehensiveUnregisteredMultiTaskCD")

    cfg_kwargs = _config_kwargs_for_dataclass(config_cls, args)
    cfg_kwargs["use_fallback_encoder_if_missing"] = False

    # If the trustH model supports this config, pass it through.
    if "trust_provided_homography" in set(getattr(config_cls, "__annotations__", {}).keys()):
        cfg_kwargs["trust_provided_homography"] = args.trust_provided_homography

    cfg = config_cls(**cfg_kwargs)

    encoder = DINOOnlyEncoder(
        fpn_channels=args.fpn_channels,
        dino_weight=args.dino_weight,
        device=str(device),
        extract_ids=args.extract_ids,
        dino_out_dim=args.dino_out_dim,
        freeze_dino=args.freeze_dino,
    ).to(device)

    model = model_cls(cfg=cfg, encoder=encoder).to(device)
    return model, cfg, module


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: nn.Module, checkpoint_path: Union[str, Path], device: torch.device, strict: bool = True) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    state = _strip_module_prefix(state)
    model_state = model.state_dict()

    if strict:
        missing, unexpected = model.load_state_dict(state, strict=True)
        print(f"[Info] strict checkpoint loaded. missing={len(missing)}, unexpected={len(unexpected)}")
        return ckpt if isinstance(ckpt, dict) else {}

    # Non-strict with shape filtering.
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[Warn] non-strict checkpoint loaded.")
    print(f"[Warn] loaded tensors={len(filtered)}, skipped tensors={len(skipped)}")
    print(f"[Warn] missing tensors={len(missing)}, unexpected tensors={len(unexpected)}")
    if skipped:
        print(f"[Warn] first skipped tensors: {skipped[:20]}")
    return ckpt if isinstance(ckpt, dict) else {}


# =============================================================================
# Metrics
# =============================================================================

def resize_target_if_needed(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.shape[-2:] == logits.shape[-2:]:
        return target.long()
    return F.interpolate(target.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / (den + 1e-6)


@torch.no_grad()
def update_confusion_matrix(
    conf: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    pred/target: [B,H,W]
    conf[row=true, col=pred]
    """
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    if valid.sum() < 1:
        return conf
    t = target[valid].long()
    p = pred[valid].long().clamp(0, num_classes - 1)
    idx = t * num_classes + p
    hist = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    conf += hist.to(conf.device)
    return conf


@torch.no_grad()
def metrics_from_confusion(conf: torch.Tensor, prefix: str = "bcd") -> Dict[str, float]:
    conf = conf.float()
    num_classes = conf.shape[0]
    out: Dict[str, float] = {}

    tp = torch.diag(conf)
    row_sum = conf.sum(dim=1)
    col_sum = conf.sum(dim=0)
    total = conf.sum().clamp_min(1.0)

    ious = tp / (row_sum + col_sum - tp + 1e-6)
    recalls = tp / (row_sum + 1e-6)
    precisions = tp / (col_sum + 1e-6)

    acc = tp.sum() / total
    out[f"{prefix}_overall_acc"] = float(acc.cpu())

    valid_classes = row_sum > 0
    if valid_classes.any():
        out[f"{prefix}_miou_all_present"] = float(ious[valid_classes].mean().cpu())
    else:
        out[f"{prefix}_miou_all_present"] = 0.0

    for c in range(num_classes):
        name = BCD_CLASS_NAMES.get(c, f"class{c}")
        out[f"{prefix}_{name}_iou"] = float(ious[c].cpu())
        out[f"{prefix}_{name}_precision"] = float(precisions[c].cpu())
        out[f"{prefix}_{name}_recall"] = float(recalls[c].cpu())

    return out


@torch.no_grad()
def binary_change_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    unknown_index: int = 2,
    prefix: str = "bcd",
) -> Dict[str, float]:
    """
    Binary changed-vs-unchanged metrics.
    target=2/255 are ignored.
    """
    target = resize_target_if_needed(target, logits)
    pred = torch.argmax(logits, dim=1)

    valid = ((target == 0) | (target == 1)) & (target != ignore_index) & (target != unknown_index)
    if valid.float().sum() < 1:
        return {
            f"{prefix}_precision": 0.0,
            f"{prefix}_recall": 0.0,
            f"{prefix}_f1": 0.0,
            f"{prefix}_iou": 0.0,
        }

    pred_v = pred[valid]
    target_v = target[valid]
    tp = ((pred_v == 1) & (target_v == 1)).sum().float()
    fp = ((pred_v == 1) & (target_v == 0)).sum().float()
    fn = ((pred_v != 1) & (target_v == 1)).sum().float()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)

    return {
        f"{prefix}_precision": float(precision.cpu()),
        f"{prefix}_recall": float(recall.cpu()),
        f"{prefix}_f1": float(f1.cpu()),
        f"{prefix}_iou": float(iou.cpu()),
    }


# =============================================================================
# Prediction saving
# =============================================================================

def colorize_bcd_np(label: np.ndarray) -> np.ndarray:
    """
    RGB visualization:
    0 unchanged -> black
    1 changed   -> white
    2 unknown   -> blue
    255 ignore   -> red
    """
    label = label.astype(np.uint8)
    vis = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    vis[label == 0] = (0, 0, 0)
    vis[label == 1] = (255, 255, 255)
    vis[label == 2] = (0, 0, 255)
    vis[label == 255] = (255, 0, 0)
    return vis


def save_prediction(
    pred: torch.Tensor,
    target: torch.Tensor,
    names: Sequence[str],
    out_dir: Path,
    save_target: bool = True,
) -> None:
    out_raw = out_dir / "pred_raw"
    out_vis = out_dir / "pred_vis"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_vis.mkdir(parents=True, exist_ok=True)

    if save_target:
        tgt_raw = out_dir / "target_raw"
        tgt_vis = out_dir / "target_vis"
        tgt_raw.mkdir(parents=True, exist_ok=True)
        tgt_vis.mkdir(parents=True, exist_ok=True)

    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    target_np = target.detach().cpu().numpy().astype(np.uint8)

    for i, name in enumerate(names):
        stem = Path(str(name)).stem
        Image.fromarray(pred_np[i]).save(out_raw / f"{stem}.png")
        Image.fromarray(colorize_bcd_np(pred_np[i])).save(out_vis / f"{stem}.png")
        if save_target:
            Image.fromarray(target_np[i]).save(tgt_raw / f"{stem}.png")
            Image.fromarray(colorize_bcd_np(target_np[i])).save(tgt_vis / f"{stem}.png")



def tensor_image_to_uint8_np(x: torch.Tensor) -> np.ndarray:
    """
    Convert RGB tensor [B,3,H,W] or [3,H,W] in [0,1] to uint8 RGB numpy.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x_np = x.detach().cpu().float().clamp(0.0, 1.0).numpy()
    x_np = np.transpose(x_np, (0, 2, 3, 1))
    return (x_np * 255.0).round().astype(np.uint8)


def normalize_dataset_type(dataset_type: str, num_classes: int) -> str:
    """
    Resolve dataset type for semantic visualization.

    Supported values:
    - auto   : choose cc_scd5 when num_classes == 5, otherwise sc_scd7 when num_classes == 7
    - cc_scd5: use CC_SCD5_PALETTE
    - sc_scd7: use SC_SCD7_PALETTE
    """
    dt = str(dataset_type or "auto").strip().lower().replace("-", "_")
    alias = {
        "ccscd5": "cc_scd5",
        "cc_scd5": "cc_scd5",
        "cc_scd_5": "cc_scd5",
        "5": "cc_scd5",
        "scscd7": "sc_scd7",
        "sc_scd7": "sc_scd7",
        "sc_scd_7": "sc_scd7",
        "7": "sc_scd7",
        "auto": "auto",
    }
    dt = alias.get(dt, dt)
    if dt == "auto":
        if int(num_classes) == len(CC_SCD5_CLASSES):
            return "cc_scd5"
        if int(num_classes) == len(SC_SCD7_CLASSES):
            return "sc_scd7"
        # For uncommon class counts, fall back to SC-SCD7 colors plus deterministic extensions.
        return "sc_scd7"
    if dt not in {"cc_scd5", "sc_scd7"}:
        raise ValueError(f"Unsupported dataset_type={dataset_type}. Use auto, cc_scd5, or sc_scd7.")
    return dt


def make_seg_palette(num_classes: int, dataset_type: str = "auto") -> np.ndarray:
    """
    Fixed RGB palette for semantic labels.
    Class 255 is handled separately as red.
    """
    dt = normalize_dataset_type(dataset_type, num_classes)
    base = CC_SCD5_PALETTE if dt == "cc_scd5" else SC_SCD7_PALETTE

    num_classes = int(num_classes)
    if num_classes <= len(base):
        return base[:num_classes].copy()

    # This extension is deterministic because the seed is fixed.
    rng = np.random.default_rng(12345)
    extra = rng.integers(0, 256, size=(num_classes - len(base), 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0)


def get_seg_class_names(num_classes: int, dataset_type: str = "auto") -> List[str]:
    """Return semantic class names matching the selected dataset palette."""
    dt = normalize_dataset_type(dataset_type, num_classes)
    base = CC_SCD5_CLASSES if dt == "cc_scd5" else SC_SCD7_CLASSES
    if num_classes <= len(base):
        return base[:num_classes]
    return base + [f"Class_{i}" for i in range(len(base), num_classes)]


def colorize_seg_np(label: np.ndarray, num_classes: int, dataset_type: str = "auto") -> np.ndarray:
    """
    RGB visualization for semantic segmentation labels.
    255 ignore -> red.
    """
    label = label.astype(np.uint8)
    palette = make_seg_palette(num_classes, dataset_type=dataset_type)
    vis = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for c in range(int(num_classes)):
        vis[label == c] = palette[c]
    vis[label == 255] = (255, 0, 0)
    return vis


def colorize_scd_np(label: np.ndarray) -> np.ndarray:
    """
    RGB visualization for SCD labels:
    0 unchanged -> black
    1 changed   -> white
    255 unknown/ignore -> red
    """
    label = label.astype(np.uint8)
    vis = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    vis[label == 0] = (0, 0, 0)
    vis[label == 1] = (255, 255, 255)
    vis[label == 255] = (255, 0, 0)
    return vis


def save_input_images(
    t1: torch.Tensor,
    t2: torch.Tensor,
    names: Sequence[str],
    out_dir: Path,
) -> None:
    """Save input T1/T2 RGB crops for visual checking."""
    t1_dir = out_dir / "input_prev_img_T1"
    t2_dir = out_dir / "input_curr_img_T2"
    t1_dir.mkdir(parents=True, exist_ok=True)
    t2_dir.mkdir(parents=True, exist_ok=True)
    t1_np = tensor_image_to_uint8_np(t1)
    t2_np = tensor_image_to_uint8_np(t2)
    for i, name in enumerate(names):
        stem = Path(str(name)).stem
        Image.fromarray(t1_np[i]).save(t1_dir / f"{stem}.png")
        Image.fromarray(t2_np[i]).save(t2_dir / f"{stem}.png")


def save_segmentation_predictions(
    out: Dict[str, Any],
    batch: Dict[str, Any],
    names: Sequence[str],
    out_dir: Path,
    num_classes: int,
    dataset_type: str = "auto",
    save_target: bool = True,
) -> None:
    """
    Save semantic segmentation results for multi/seg/scd modes.

    Saved folders:
      seg_A_pred_raw / seg_A_pred_vis               : T1 semantic prediction from seg_A
      seg_B_pred_raw / seg_B_pred_vis               : T2 semantic prediction from seg_B
      seg_A_to_T2_pred_raw / seg_A_to_T2_pred_vis   : T1 semantic prediction warped to T2, if present
      seg_A_target_raw / seg_A_target_vis           : prev_seg GT
      seg_B_target_raw / seg_B_target_vis           : curr_seg GT
    """
    pred_items: List[Tuple[str, torch.Tensor]] = []
    if "seg_A" in out and torch.is_tensor(out["seg_A"]):
        pred_items.append(("seg_A", torch.argmax(out["seg_A"], dim=1)))
    if "seg_B" in out and torch.is_tensor(out["seg_B"]):
        pred_items.append(("seg_B", torch.argmax(out["seg_B"], dim=1)))
    if "seg_A_to_T2" in out and torch.is_tensor(out["seg_A_to_T2"]):
        pred_items.append(("seg_A_to_T2", torch.argmax(out["seg_A_to_T2"], dim=1)))

    for key, pred in pred_items:
        raw_dir = out_dir / f"{key}_pred_raw"
        vis_dir = out_dir / f"{key}_pred_vis"
        raw_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        pred_np = pred.detach().cpu().numpy().astype(np.uint8)
        for i, name in enumerate(names):
            stem = Path(str(name)).stem
            Image.fromarray(pred_np[i]).save(raw_dir / f"{stem}.png")
            Image.fromarray(colorize_seg_np(pred_np[i], num_classes, dataset_type=dataset_type)).save(vis_dir / f"{stem}.png")

    if not save_target:
        return

    target_map = {
        "seg_A_target": batch.get("label_a", None),
        "seg_B_target": batch.get("label_b", None),
    }
    for key, target in target_map.items():
        if not torch.is_tensor(target):
            continue
        raw_dir = out_dir / f"{key}_raw"
        vis_dir = out_dir / f"{key}_vis"
        raw_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        target_np = target.detach().cpu().numpy().astype(np.uint8)
        for i, name in enumerate(names):
            stem = Path(str(name)).stem
            Image.fromarray(target_np[i]).save(raw_dir / f"{stem}.png")
            Image.fromarray(colorize_seg_np(target_np[i], num_classes, dataset_type=dataset_type)).save(vis_dir / f"{stem}.png")


def save_scd_predictions(
    out: Dict[str, Any],
    batch: Dict[str, Any],
    names: Sequence[str],
    out_dir: Path,
    save_target: bool = True,
    ignore_index: int = 255,
    unknown_index: int = 2,
) -> None:
    """
    Save SCD prediction if model outputs SCD_logits or SCD.
    Target is derived from BCD: 0/1 keep, 2/255 -> 255.
    """
    pred: Optional[torch.Tensor] = None
    if "SCD_logits" in out and torch.is_tensor(out["SCD_logits"]):
        pred = torch.argmax(out["SCD_logits"], dim=1)
    elif "SCD" in out and torch.is_tensor(out["SCD"]):
        pred = out["SCD"].long()
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred[:, 0]

    if pred is None:
        return

    raw_dir = out_dir / "scd_pred_raw"
    vis_dir = out_dir / "scd_pred_vis"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    for i, name in enumerate(names):
        stem = Path(str(name)).stem
        Image.fromarray(pred_np[i]).save(raw_dir / f"{stem}.png")
        Image.fromarray(colorize_scd_np(pred_np[i])).save(vis_dir / f"{stem}.png")

    if not save_target or "bcd" not in batch or not torch.is_tensor(batch["bcd"]):
        return

    target = batch["bcd"].clone().long()
    target[(target == unknown_index) | (target == ignore_index)] = 255
    target[(target != 0) & (target != 1) & (target != 255)] = 255

    tgt_raw_dir = out_dir / "scd_target_raw"
    tgt_vis_dir = out_dir / "scd_target_vis"
    tgt_raw_dir.mkdir(parents=True, exist_ok=True)
    tgt_vis_dir.mkdir(parents=True, exist_ok=True)
    target_np = target.detach().cpu().numpy().astype(np.uint8)
    for i, name in enumerate(names):
        stem = Path(str(name)).stem
        Image.fromarray(target_np[i]).save(tgt_raw_dir / f"{stem}.png")
        Image.fromarray(colorize_scd_np(target_np[i])).save(tgt_vis_dir / f"{stem}.png")

def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv_row(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(data.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(data)


# =============================================================================
# Test loop
# =============================================================================

@torch.no_grad()
def run_test(
    model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    total_binary: Dict[str, float] = {}
    num_steps = 0
    conf = torch.zeros(args.bcd_num_classes, args.bcd_num_classes, dtype=torch.long, device=device)

    out_dir = Path(args.output_dir)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        H0_t1_to_t2 = None
        if args.use_provided_homography:
            if "H0_t1_to_t2" not in batch:
                raise KeyError("Dataset did not return H0_t1_to_t2, but --use_provided_homography is enabled.")
            has_H0 = batch.get("has_H0", None)
            if has_H0 is not None and (not bool(has_H0.bool().all().item())) and (not args.allow_missing_homography):
                raise RuntimeError("Some samples in the batch do not have provided homography.")
            H0_t1_to_t2 = batch["H0_t1_to_t2"]

        out = model(
            batch["t1"],
            batch["t2"],
            H0_t1_to_t2=H0_t1_to_t2,
            task=args.task,
        )

        if "BCD" not in out:
            raise KeyError("Model output does not contain 'BCD'. Use --task bcd or --task multi.")

        logits = out["BCD"]
        target = resize_target_if_needed(batch["bcd"], logits)
        pred = torch.argmax(logits, dim=1)

        # Three-class confusion metrics include class 2.
        conf = update_confusion_matrix(
            conf=conf,
            pred=pred,
            target=target,
            num_classes=args.bcd_num_classes,
            ignore_index=args.bcd_ignore_index,
        )

        # Binary change metrics ignore target=2/255.
        binary = binary_change_metrics_from_logits(
            logits=logits,
            target=target,
            ignore_index=args.bcd_ignore_index,
            unknown_index=args.unknown_index,
            prefix="bcd",
        )
        for k, v in binary.items():
            total_binary[k] = total_binary.get(k, 0.0) + float(v)

        names = batch.get("name", [f"sample_{step:06d}_{i}" for i in range(pred.shape[0])])
        if isinstance(names, str):
            names = [names]

        if args.save_pred:
            save_prediction(pred, target, names, out_dir, save_target=args.save_target)

        if args.save_inputs:
            save_input_images(batch["t1"], batch["t2"], names, out_dir)

        if args.save_seg and args.task in {"seg", "scd", "multi"}:
            save_segmentation_predictions(
                out=out,
                batch=batch,
                names=names,
                out_dir=out_dir,
                num_classes=args.seg_num_classes,
                dataset_type=args.dataset_type,
                save_target=args.save_target,
            )

        if args.save_scd and args.task in {"scd", "multi"}:
            save_scd_predictions(
                out=out,
                batch=batch,
                names=names,
                out_dir=out_dir,
                save_target=args.save_target,
                ignore_index=args.bcd_ignore_index,
                unknown_index=args.unknown_index,
            )

        if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == len(loader)):
            msg = " ".join(f"{k}={v:.4f}" for k, v in binary.items())
            print(f"[Test][{step:04d}/{len(loader):04d}] {msg}")

        num_steps += 1

    if num_steps < 1:
        raise RuntimeError("No test batches were processed.")

    binary_avg = {k: v / num_steps for k, v in total_binary.items()}
    multi = metrics_from_confusion(conf, prefix="bcd3")

    result: Dict[str, Any] = {}
    result.update(binary_avg)
    result.update(multi)
    result["num_batches"] = int(num_steps)
    result["num_samples"] = int(len(loader.dataset))
    result["confusion_matrix"] = conf.detach().cpu().tolist()

    return result


# =============================================================================
# DataLoader
# =============================================================================

def make_test_dataset(args: argparse.Namespace) -> CCSCD5PairTestDataset:
    return CCSCD5PairTestDataset(
        data_root=args.data_root,
        bcd_dir=args.bcd_dir,
        curr_img_dir=args.curr_img_dir,
        curr_seg_dir=args.curr_seg_dir,
        prev_img_dir=args.prev_img_dir,
        prev_seg_dir=args.prev_seg_dir,
        bcd_label_mode=args.bcd_label_mode,
        seg_label_mode=args.seg_label_mode,
        seg_ignore_value=args.seg_ignore_value,
        seg_num_classes=args.seg_num_classes,
        use_provided_homography=args.use_provided_homography,
        homography_dir=args.homography_dir,
        homography_csv=args.homography_csv,
        allow_missing_homography=args.allow_missing_homography,
    )


def make_test_loader(args: argparse.Namespace) -> Tuple[CCSCD5PairTestDataset, DataLoader]:
    dataset = make_test_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    return dataset, loader


# =============================================================================
# Argument parser
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test 3-class BCD model with optional trusted homography.")

    # Data.
    parser.add_argument("--data_root", type=str, default="./DATA/SCSCD7_3/train(256_256)")
    parser.add_argument("--output_dir", type=str, default="./test_outputs/SCSCD7_3/test/")  # 输出权重路径
    parser.add_argument("--dataset_type", type=str, default="SCSCD7", choices=["auto", "cc_scd5", "sc_scd7", "CCSCD5", "SCSCD7"], help="Semantic dataset palette: auto, cc_scd5, or sc_scd7.")
    parser.add_argument("--bcd_dir", type=str, default="bcd")


    return parser


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] device = {device}")
    print(f"[Info] model_module = {args.model_module}")
    print(f"[Info] checkpoint = {args.checkpoint}")
    print(f"[Info] data_root = {args.data_root}")
    resolved_dataset_type = normalize_dataset_type(args.dataset_type, args.seg_num_classes)
    seg_class_names = get_seg_class_names(args.seg_num_classes, resolved_dataset_type)
    print(f"[Info] dataset_type = {args.dataset_type} -> {resolved_dataset_type}")
    print(f"[Info] seg class names = {seg_class_names}")
    print(f"[Info] seg palette = {make_seg_palette(args.seg_num_classes, resolved_dataset_type).tolist()}")
    print(f"[Info] bcd_num_classes = {args.bcd_num_classes}")
    print(f"[Info] bcd_label_mode = {args.bcd_label_mode}")
    print(f"[Info] use_provided_homography = {args.use_provided_homography}")
    print(f"[Info] trust_provided_homography = {args.trust_provided_homography}")
    print(f"[Info] save_pred = {args.save_pred}, save_seg = {args.save_seg}, save_scd = {args.save_scd}, save_inputs = {args.save_inputs}")

    dataset, loader = make_test_loader(args)

    print(f"[Info] valid paired test samples = {len(dataset)}")
    if len(dataset.missing) > 0:
        print(f"[Warn] skipped samples with missing image/label files = {len(dataset.missing)}")
        print(f"[Warn] first missing stems = {dataset.missing[:10]}")
    if getattr(dataset, "missing_h", None):
        if len(dataset.missing_h) > 0:
            print(f"[Warn] skipped samples with missing homography = {len(dataset.missing_h)}")
            print(f"[Warn] first missing H stems = {dataset.missing_h[:10]}")

    sample0 = dataset[0]
    print(f"[Debug] first sample = {sample0['name']}")
    print(f"[Debug] T1 shape = {tuple(sample0['t1'].shape)}, T2 shape = {tuple(sample0['t2'].shape)}")
    print(f"[Debug] BCD unique = {torch.unique(sample0['bcd']).tolist()}")
    if args.use_provided_homography:
        print(f"[Debug] H0_t1_to_t2 =\n{sample0['H0_t1_to_t2']}")

    model, cfg, module = build_model(args, device)
    ckpt = load_checkpoint(model, args.checkpoint, device=device, strict=args.strict_load)
    model.eval()

    metrics = run_test(model, loader, args, device)

    save_json(metrics, out_dir / "metrics.json")
    flat_metrics = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    save_csv_row(flat_metrics, out_dir / "metrics.csv")

    print("\n" + "=" * 80)
    print("[Test finished]")
    for k, v in flat_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("confusion_matrix:")
    print(np.asarray(metrics["confusion_matrix"], dtype=np.int64))
    print(f"[Info] saved metrics to: {out_dir / 'metrics.json'}")
    if args.save_pred or args.save_seg or args.save_scd or args.save_inputs:
        print(f"[Info] saved visual outputs to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
