
# -*- coding: utf-8 -*-
"""
train_ccscd5_dataset_diffreg_stabilized.py

Stabilized training script for model_3_6_stabilized_differentiable_registration.py.

Designed as a clean comparison training file for the rebuilt model:
- BCD loss is computed only inside the model-provided overlap-used/core mask.
- Trainable SCD head is supervised with the binary change label when available.
- Risky modules are disabled by default: traditional residual refinement,
  residual flow refinement, and semantic guidance for BCD.
- Registration auxiliary losses are zero by default, so real changes are not
  accidentally punished as feature inconsistency.
- The script keeps the original CC-SCD5/SCSCD7 folder convention.

Expected directory layout:
    data_root/
        bcd/                 # BCD labels; for 3-class training use pixel values 0/1/2
        curr_img/
        curr_seg/
        prev_img/
        prev_seg/
        # optional, recommended for random-offset crops:
        homography/           # one <stem>.npy/.txt per sample, each [3,3] H_t1_to_t2
        crop_meta.csv         # alternatively stores start_x/start_y/start_x2/start_y2 or shift_x/shift_y

Model forward convention:
    T1 = prev_img, label_a = prev_seg
    T2 = curr_img, label_b = curr_seg
    Outputs are in T2 coordinates by default.

For 3-class BCD labels:
    0 = unchanged
    1 = changed
    2 = unknown / non-overlap

If the dataset was created by random offset cropping, pass the true H_t1_to_t2 to the
model with --use_provided_homography. This keeps the model-computed overlap consistent
with the class-2 area in the BCD target.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import random
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


# =============================================================================
# Dataset utilities
# =============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

CC_SCD5_CLASSES = ["Others", "Water", "Building", "Vegetation", "Road"]
SC_SCD7_CLASSES = ["Bareland", "Water", "Building", "Structure", "Farmland", "Vegetation", "Road"]
BCD_CLASSES = ["Unchanged", "Changed area", "Unknown/non-overlap"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _read_bcd(path: Path, mode: str = "binary_0_255") -> torch.Tensor:
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


def _normalize_csv_row(row: Dict[str, Any]) -> Dict[str, str]:
    """Normalize CSV keys for flexible crop metadata parsing."""
    out: Dict[str, str] = {}
    for k, v in row.items():
        if k is None:
            continue
        out[str(k).strip().lower()] = "" if v is None else str(v).strip()
    return out


def _get_csv_value(row: Dict[str, str], *keys: str) -> Optional[str]:
    for key in keys:
        key_l = key.lower()
        if key_l in row and row[key_l] != "":
            return row[key_l]
    return None


def _get_csv_float(row: Dict[str, str], *keys: str) -> Optional[float]:
    value = _get_csv_value(row, *keys)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _homography_from_offsets(shift_x: float, shift_y: float) -> np.ndarray:
    """
    Build H mapping T1 crop coordinates to T2/curr crop coordinates.

    In the data maker, if:
        start_x2 = start_x + shift_x
        start_y2 = start_y + shift_y

    then a point in T1 crop coordinates maps to T2 crop coordinates as:
        x_t2 = x_t1 - shift_x
        y_t2 = y_t1 - shift_y
    """
    return np.array(
        [[1.0, 0.0, -float(shift_x)],
         [0.0, 1.0, -float(shift_y)],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _homography_from_crop_positions(start_x: float, start_y: float, start_x2: float, start_y2: float) -> np.ndarray:
    shift_x = float(start_x2) - float(start_x)
    shift_y = float(start_y2) - float(start_y)
    return _homography_from_offsets(shift_x=shift_x, shift_y=shift_y)


def _load_homography_matrix(path: Path) -> np.ndarray:
    """Load one [3,3] homography from .npy, .txt, or .csv."""
    if path.suffix.lower() == ".npy":
        H = np.load(path)
    else:
        try:
            H = np.loadtxt(path, delimiter=",")
        except Exception:
            H = np.loadtxt(path)
    H = np.asarray(H, dtype=np.float32)
    if H.shape == (1, 9):
        H = H.reshape(3, 3)
    if H.shape == (9,):
        H = H.reshape(3, 3)
    if H.shape != (3, 3):
        raise ValueError(f"Homography file must contain a [3,3] matrix, got {H.shape}: {path}")
    return H


def _find_homography_file(folder: Path, stem: str) -> Optional[Path]:
    for ext in (".npy", ".txt", ".csv"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _load_homography_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Load homographies from a metadata CSV. Supported columns include either:
    - name/stem/filename/save_name plus h00,h01,...,h22; or
    - name/stem/filename/save_name plus shift_x,shift_y; or
    - name/stem/filename/save_name plus start_x,start_y,start_x2,start_y2.
    """
    out: Dict[str, np.ndarray] = {}
    if not csv_path.exists():
        return out

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = _normalize_csv_row(raw)
            name = _get_csv_value(row, "stem", "name", "filename", "file", "save_name", "sample", "id")
            if not name:
                continue
            stem = Path(name).stem

            h_keys = [
                ("h00", "h01", "h02"),
                ("h10", "h11", "h12"),
                ("h20", "h21", "h22"),
            ]
            h_vals: List[float] = []
            has_full_h = True
            for row_keys in h_keys:
                for key in row_keys:
                    value = _get_csv_float(row, key, key.replace("h", "h_"), key.upper())
                    if value is None:
                        has_full_h = False
                        break
                    h_vals.append(value)
                if not has_full_h:
                    break
            if has_full_h and len(h_vals) == 9:
                out[stem] = np.asarray(h_vals, dtype=np.float32).reshape(3, 3)
                continue

            shift_x = _get_csv_float(row, "shift_x", "dx", "offset_x")
            shift_y = _get_csv_float(row, "shift_y", "dy", "offset_y")
            if shift_x is not None and shift_y is not None:
                out[stem] = _homography_from_offsets(shift_x=shift_x, shift_y=shift_y)
                continue

            start_x = _get_csv_float(row, "start_x", "x1", "prev_x", "t1_x")
            start_y = _get_csv_float(row, "start_y", "y1", "prev_y", "t1_y")
            start_x2 = _get_csv_float(row, "start_x2", "x2", "curr_x", "t2_x")
            start_y2 = _get_csv_float(row, "start_y2", "y2", "curr_y", "t2_y")
            if None not in (start_x, start_y, start_x2, start_y2):
                out[stem] = _homography_from_crop_positions(
                    start_x=float(start_x),
                    start_y=float(start_y),
                    start_x2=float(start_x2),
                    start_y2=float(start_y2),
                )
                continue

    return out


class CCSCD5PairDataset(Dataset):
    """Dataset that pairs files by the same filename stem."""

    def __init__(
        self,
        data_root: Union[str, Path],
        bcd_dir: str = "bcd",
        curr_img_dir: str = "curr_img",
        curr_seg_dir: str = "curr_seg",
        prev_img_dir: str = "prev_img",
        prev_seg_dir: str = "prev_seg",
        bcd_label_mode: str = "binary_0_255",
        seg_label_mode: str = "grayscale",
        seg_ignore_value: int = 255,
        seg_num_classes: int = 7,
        use_provided_homography: bool = False,
        homography_dir: str = "homography",
        homography_csv: str = "crop_meta.csv",
        require_homography: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.bcd_label_mode = bcd_label_mode
        self.seg_label_mode = seg_label_mode
        self.seg_ignore_value = seg_ignore_value
        self.seg_num_classes = seg_num_classes
        self.use_provided_homography = bool(use_provided_homography)
        self.homography_dir = self.data_root / homography_dir
        self.homography_csv = self.data_root / homography_csv if homography_csv else Path("")
        self.require_homography = bool(require_homography)
        self.homography_by_stem: Dict[str, np.ndarray] = (
            _load_homography_csv(self.homography_csv)
            if self.use_provided_homography and homography_csv
            else {}
        )
        self.missing_homography: List[str] = []

        bcd_files = _list_image_files(self.data_root / bcd_dir)
        curr_img = _index_by_stem(_list_image_files(self.data_root / curr_img_dir))
        curr_seg = _index_by_stem(_list_image_files(self.data_root / curr_seg_dir))
        prev_img = _index_by_stem(_list_image_files(self.data_root / prev_img_dir))
        prev_seg = _index_by_stem(_list_image_files(self.data_root / prev_seg_dir))

        samples: List[Dict[str, Path]] = []
        missing: List[str] = []
        for bcd_path in bcd_files:
            key = bcd_path.stem
            paths: Dict[str, Optional[Path]] = {
                "bcd": bcd_path,
                "curr_img": curr_img.get(key),
                "curr_seg": curr_seg.get(key),
                "prev_img": prev_img.get(key),
                "prev_seg": prev_seg.get(key),
            }
            if all(v is not None for v in paths.values()):
                samples.append(paths)  # type: ignore[arg-type]
            else:
                missing.append(key)

        if not samples:
            raise RuntimeError(
                "No valid samples found. Files must share the same stem across "
                "bcd/curr_img/curr_seg/prev_img/prev_seg."
            )
        self.samples = samples
        self.missing = missing

        if self.use_provided_homography and self.require_homography:
            self.missing_homography = [
                s["bcd"].stem for s in self.samples
                if not self._has_homography(s["bcd"].stem)
            ]
            if self.missing_homography:
                raise RuntimeError(
                    "--use_provided_homography is enabled but some samples have no H_t1_to_t2. "
                    f"Missing count={len(self.missing_homography)}. First missing={self.missing_homography[:10]}. "
                    f"Expected either {self.homography_dir}/<stem>.npy/.txt/.csv or metadata CSV {self.homography_csv}."
                )

    def _has_homography(self, stem: str) -> bool:
        if stem in self.homography_by_stem:
            return True
        return _find_homography_file(self.homography_dir, stem) is not None

    def _read_homography(self, stem: str) -> np.ndarray:
        if stem in self.homography_by_stem:
            return self.homography_by_stem[stem].astype(np.float32)
        path = _find_homography_file(self.homography_dir, stem)
        if path is not None:
            return _load_homography_matrix(path)
        if self.require_homography:
            raise FileNotFoundError(
                f"No homography found for sample '{stem}'. "
                f"Expected {self.homography_dir}/{stem}.npy/.txt/.csv or row in {self.homography_csv}."
            )
        return np.eye(3, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        item: Dict[str, Any] = {
            "t1": _read_rgb(s["prev_img"]),
            "t2": _read_rgb(s["curr_img"]),
            "bcd": _read_bcd(s["bcd"], self.bcd_label_mode),
            "label_a": _read_seg(s["prev_seg"], self.seg_label_mode, self.seg_ignore_value, self.seg_num_classes),
            "label_b": _read_seg(s["curr_seg"], self.seg_label_mode, self.seg_ignore_value, self.seg_num_classes),
            "name": s["bcd"].stem,
        }
        if self.use_provided_homography:
            H = self._read_homography(s["bcd"].stem)
            item["H0_t1_to_t2"] = torch.from_numpy(H).float()
        return item


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
# Model builder
# =============================================================================


def _config_kwargs_for_dataclass(config_cls: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Pass only args that exist in the model config dataclass."""
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


# =============================================================================
# Loss utilities
# =============================================================================


def resize_target_if_needed(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.shape[-2:] == logits.shape[-2:]:
        return target.long()
    return F.interpolate(target.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()


def binary_erode_for_loss(mask: torch.Tensor, k: int = 9, threshold: float = 0.5) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError(f"Expected mask [B,1,H,W] or [B,H,W], got {tuple(mask.shape)}")
    if k <= 1:
        return (mask > threshold).float()
    if k % 2 != 1:
        raise ValueError("Erosion kernel size must be odd.")
    mask = (mask > threshold).float()
    pad = k // 2
    inv = 1.0 - mask
    inv = F.pad(inv, (pad, pad, pad, pad), mode="constant", value=1.0)
    return (1.0 - F.max_pool2d(inv, kernel_size=k, stride=1, padding=0)).clamp(0.0, 1.0)


def get_overlap_mask(
    out: Dict[str, Any],
    logits: torch.Tensor,
    args: argparse.Namespace,
    preferred: Sequence[str],
) -> Optional[torch.Tensor]:
    mask = None
    for key in preferred:
        if key in out and torch.is_tensor(out[key]):
            mask = out[key]
            break
    if mask is None:
        for key in ["BCD_overlap_used", "BCD_overlap_core", "BCD_valid_mask_for_loss", "SCD_overlap_used"]:
            if key in out and torch.is_tensor(out[key]):
                mask = out[key]
                break
    if mask is None:
        for key in ["overlap_mask", "BCD_overlap_raw", "SCD_overlap_mask"]:
            if key in out and torch.is_tensor(out[key]):
                mask = binary_erode_for_loss(out[key].float(), k=args.overlap_erode_ks)
                break
    if mask is None:
        return None
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != logits.shape[-2:]:
        mask = F.interpolate(mask.float(), size=logits.shape[-2:], mode="nearest")
    return mask.float()


def local_prepare_change_target(
    target: torch.Tensor,
    overlap_mask: Optional[torch.Tensor],
    num_classes: int,
    ignore_index: int = 255,
    unknown_index: int = 2,
    target_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]
    out = target.clone().long()
    if target_hw is not None and out.shape[-2:] != target_hw:
        out = F.interpolate(out.unsqueeze(1).float(), size=target_hw, mode="nearest").squeeze(1).long()
    out[out == unknown_index] = ignore_index if num_classes <= 2 else unknown_index
    if overlap_mask is not None:
        if overlap_mask.dim() == 4:
            ov = overlap_mask[:, 0] > 0.5
        else:
            ov = overlap_mask > 0.5
        if ov.shape[-2:] != out.shape[-2:]:
            ov = F.interpolate(ov.float().unsqueeze(1), size=out.shape[-2:], mode="nearest")[:, 0] > 0.5
        if num_classes <= 2:
            out[~ov] = ignore_index
        else:
            out[~ov] = unknown_index
    return out


def prepare_change_target_for_loss(
    module: Any,
    target: torch.Tensor,
    overlap_mask: Optional[torch.Tensor],
    num_classes: int,
    ignore_index: int,
    unknown_index: int,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    # Prefer the helper from the rebuilt model when possible.
    if overlap_mask is not None and hasattr(module, "prepare_bcd_target_for_loss"):
        try:
            target_resized = target
            if target_resized.dim() == 4 and target_resized.shape[1] == 1:
                target_resized = target_resized[:, 0]
            if target_resized.shape[-2:] != target_hw:
                target_resized = F.interpolate(
                    target_resized.unsqueeze(1).float(), size=target_hw, mode="nearest"
                ).squeeze(1).long()
            return module.prepare_bcd_target_for_loss(
                target=target_resized,
                overlap_mask=overlap_mask,
                bcd_num_classes=num_classes,
                ignore_index=ignore_index,
                unknown_index=unknown_index,
            ).long()
        except Exception:
            pass
    return local_prepare_change_target(
        target=target,
        overlap_mask=overlap_mask,
        num_classes=num_classes,
        ignore_index=ignore_index,
        unknown_index=unknown_index,
        target_hw=target_hw,
    )


def dice_loss_changed_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    changed_index: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    if logits.size(1) <= changed_index:
        return logits.sum() * 0.0
    prob = torch.softmax(logits, dim=1)[:, changed_index]
    tgt = (target == changed_index).float()
    valid = valid_mask.float()
    prob = prob * valid
    tgt = tgt * valid
    inter = (prob * tgt).sum()
    denom = prob.sum() + tgt.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def masked_ce_dice_change_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    overlap_mask: Optional[torch.Tensor],
    num_classes: int,
    ignore_index: int = 255,
    unknown_index: int = 2,
    dice_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() != 4:
        raise ValueError(f"Expected logits [B,C,H,W], got {tuple(logits.shape)}")
    target = resize_target_if_needed(target, logits)

    if overlap_mask is not None:
        if overlap_mask.dim() == 3:
            overlap_mask = overlap_mask.unsqueeze(1)
        if overlap_mask.shape[-2:] != logits.shape[-2:]:
            overlap_mask = F.interpolate(overlap_mask.float(), size=logits.shape[-2:], mode="nearest")
        overlap_bool = overlap_mask[:, 0] > 0.5
    else:
        overlap_bool = torch.ones_like(target, dtype=torch.bool)

    if num_classes <= 2:
        valid = (target == 0) | (target == 1)
        valid = valid & overlap_bool
        safe_target = target.clone()
        safe_target[~valid] = ignore_index
    else:
        valid = ((target >= 0) & (target < num_classes)) | (target == unknown_index)
        valid = valid & (target != ignore_index)
        safe_target = target.clone()
        safe_target[~valid] = ignore_index

    valid_count = valid.float().sum()
    valid_ratio = valid.float().mean().detach()
    if valid_count < 1:
        return logits.sum() * 0.0, valid_ratio

    ce = F.cross_entropy(logits, safe_target.long(), ignore_index=ignore_index)
    # Dice focuses on the actual 0/1 overlap area. Unknown class is not part of dice.
    dice_valid = ((target == 0) | (target == 1)) & overlap_bool
    if dice_valid.float().sum() < 1 or dice_weight <= 0:
        return ce, valid_ratio
    dice = dice_loss_changed_from_logits(logits, safe_target, dice_valid, changed_index=1)
    return ce + dice_weight * dice, valid_ratio


@torch.no_grad()
def compute_change_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    overlap_mask: Optional[torch.Tensor],
    ignore_index: int = 255,
    unknown_index: int = 2,
    prefix: str = "bcd",
) -> Dict[str, float]:
    target = resize_target_if_needed(target, logits)
    pred = torch.argmax(logits, dim=1)
    valid = (target == 0) | (target == 1)
    valid = valid & (target != ignore_index) & (target != unknown_index)
    if overlap_mask is not None:
        if overlap_mask.dim() == 3:
            overlap_mask = overlap_mask.unsqueeze(1)
        if overlap_mask.shape[-2:] != logits.shape[-2:]:
            overlap_mask = F.interpolate(overlap_mask.float(), size=logits.shape[-2:], mode="nearest")
        valid = valid & (overlap_mask[:, 0] > 0.5)
    if valid.float().sum() < 1:
        return {f"{prefix}_precision": 0.0, f"{prefix}_recall": 0.0, f"{prefix}_f1": 0.0, f"{prefix}_iou": 0.0}
    pred = pred[valid]
    target = target[valid]
    tp = ((pred == 1) & (target == 1)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    # In 3-class BCD, predicting unknown=2 on a changed pixel should count as missed change.
    fn = ((pred != 1) & (target == 1)).sum().float()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {
        f"{prefix}_precision": float(precision.detach().cpu()),
        f"{prefix}_recall": float(recall.detach().cpu()),
        f"{prefix}_f1": float(f1.detach().cpu()),
        f"{prefix}_iou": float(iou.detach().cpu()),
    }


def scalar_tensor(x: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.float().mean()
    raise TypeError(type(x))


def compute_losses(
    out: Dict[str, Any],
    args: argparse.Namespace,
    module: Any,
    target_bcd: Optional[torch.Tensor],
    label_a: Optional[torch.Tensor],
    label_b: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses: Dict[str, torch.Tensor] = {}
    extra_logs: Dict[str, torch.Tensor] = {}

    # BCD loss.
    if args.task in {"bcd", "multi"} and args.lambda_bcd > 0:
        if target_bcd is None:
            raise RuntimeError("BCD training requires BCD labels.")
        if "BCD" not in out:
            raise KeyError("Model output does not contain 'BCD'.")
        logits_bcd = out["BCD"]
        bcd_overlap = get_overlap_mask(
            out,
            logits_bcd,
            args,
            preferred=["BCD_overlap_used", "BCD_overlap_core", "BCD_valid_mask_for_loss"],
        )
        target_for_bcd = prepare_change_target_for_loss(
            module=module,
            target=target_bcd,
            overlap_mask=bcd_overlap,
            num_classes=args.bcd_num_classes,
            ignore_index=args.bcd_ignore_index,
            unknown_index=args.unknown_index,
            target_hw=logits_bcd.shape[-2:],
        )
        loss_bcd, bcd_valid_ratio = masked_ce_dice_change_loss(
            logits=logits_bcd,
            target=target_for_bcd,
            overlap_mask=None,
            num_classes=args.bcd_num_classes,
            ignore_index=args.bcd_ignore_index,
            unknown_index=args.unknown_index,
            dice_weight=args.bcd_dice_weight,
        )
        losses["loss_bcd"] = loss_bcd
        extra_logs["bcd_valid_ratio"] = bcd_valid_ratio

    # Trainable SCD loss. The SCD head predicts binary semantic change in T2 coordinates.
    if args.task in {"scd", "multi"} and args.lambda_scd > 0 and "SCD_logits" in out:
        if target_bcd is None:
            raise RuntimeError("SCD change-head training uses BCD-style change labels.")
        logits_scd = out["SCD_logits"]
        scd_overlap = get_overlap_mask(out, logits_scd, args, preferred=["SCD_overlap_used"])
        target_for_scd = prepare_change_target_for_loss(
            module=module,
            target=target_bcd,
            overlap_mask=scd_overlap,
            num_classes=2,
            ignore_index=args.bcd_ignore_index,
            unknown_index=args.unknown_index,
            target_hw=logits_scd.shape[-2:],
        )
        loss_scd, scd_valid_ratio = masked_ce_dice_change_loss(
            logits=logits_scd,
            target=target_for_scd,
            overlap_mask=None,
            num_classes=2,
            ignore_index=args.bcd_ignore_index,
            unknown_index=args.unknown_index,
            dice_weight=args.scd_dice_weight,
        )
        losses["loss_scd"] = loss_scd
        extra_logs["scd_valid_ratio"] = scd_valid_ratio

    # Semantic segmentation loss for both dates.
    if args.task in {"seg", "scd", "multi"} and args.lambda_seg > 0:
        if label_a is None or label_b is None:
            raise RuntimeError("Semantic training requires prev_seg and curr_seg labels.")
        if "seg_A" not in out or "seg_B" not in out:
            raise KeyError("Model output must contain 'seg_A' and 'seg_B'.")
        target_a = resize_target_if_needed(label_a, out["seg_A"])
        target_b = resize_target_if_needed(label_b, out["seg_B"])
        losses["loss_seg_A"] = F.cross_entropy(out["seg_A"], target_a.long(), ignore_index=255)
        losses["loss_seg_B"] = F.cross_entropy(out["seg_B"], target_b.long(), ignore_index=255)

    # Registration auxiliary terms. Defaults are zero because these terms can punish real changes.
    if args.lambda_reg_feature > 0 and "reg_feature_consistency_loss" in out:
        losses["loss_reg_feature"] = scalar_tensor(out["reg_feature_consistency_loss"])
    if args.lambda_reg_flow_smooth > 0 and "reg_flow_smoothness_loss" in out:
        losses["loss_reg_flow_smooth"] = scalar_tensor(out["reg_flow_smoothness_loss"])
    if args.lambda_reg_flow_mag > 0 and "reg_flow_magnitude_loss" in out:
        losses["loss_reg_flow_mag"] = scalar_tensor(out["reg_flow_magnitude_loss"])
    if args.lambda_reg_affine > 0 and "reg_affine_param_l2_loss" in out:
        losses["loss_reg_affine"] = scalar_tensor(out["reg_affine_param_l2_loss"])

    if not losses:
        raise RuntimeError("No loss was computed. Check --task and lambda weights.")

    device = next(v.device for v in losses.values())
    total = torch.zeros((), device=device)
    if "loss_bcd" in losses:
        total = total + args.lambda_bcd * losses["loss_bcd"]
    if "loss_scd" in losses:
        total = total + args.lambda_scd * losses["loss_scd"]
    if "loss_seg_A" in losses:
        total = total + args.lambda_seg * (losses["loss_seg_A"] + losses["loss_seg_B"])
    if "loss_reg_feature" in losses:
        total = total + args.lambda_reg_feature * losses["loss_reg_feature"]
    if "loss_reg_flow_smooth" in losses:
        total = total + args.lambda_reg_flow_smooth * losses["loss_reg_flow_smooth"]
    if "loss_reg_flow_mag" in losses:
        total = total + args.lambda_reg_flow_mag * losses["loss_reg_flow_mag"]
    if "loss_reg_affine" in losses:
        total = total + args.lambda_reg_affine * losses["loss_reg_affine"]

    log = {k: float(v.detach().cpu()) for k, v in losses.items()}
    log.update({k: float(v.detach().cpu()) for k, v in extra_logs.items()})
    log["loss_total"] = float(total.detach().cpu())
    return total, log


# =============================================================================
# Metrics and logging helpers
# =============================================================================


def safe_sum_lengths(x: Any) -> int:
    if isinstance(x, list):
        total = 0
        for item in x:
            try:
                total += len(item)
            except TypeError:
                pass
        return total
    try:
        return len(x)
    except TypeError:
        return 0


def tensor_mean_float(x: Any, default: float = 0.0) -> float:
    if torch.is_tensor(x):
        if x.numel() == 0:
            return default
        return float(x.float().mean().detach().cpu())
    try:
        return float(x)
    except Exception:
        return default


def bool_list_ratio(x: Any) -> float:
    if isinstance(x, list) and len(x) > 0:
        vals = [1.0 if bool(v) else 0.0 for v in x]
        return float(sum(vals) / max(len(vals), 1))
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    return 0.0


def to_plain_number(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (int, float, str, bool)):
        return v
    if hasattr(v, "detach"):
        v = v.detach().cpu()
        if v.numel() == 1:
            return float(v.item())
        return str(v.tolist())
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except Exception:
            return str(v)
    return str(v)


def resolve_metrics_csv_path(args: argparse.Namespace) -> Path:
    if getattr(args, "metrics_csv", ""):
        return Path(args.metrics_csv)
    return Path(args.save_dir) / "epoch_metrics.csv"


def load_epoch_metrics_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_epoch_metrics_csv(history: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    preferred = ["epoch", "lr", "is_best", "best_metric", "checkpoint_path"]
    keys = set()
    for row in history:
        keys.update(row.keys())
    metric_cols = sorted(k for k in keys if k not in preferred)
    fieldnames = [k for k in preferred if k in keys] + metric_cols
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(csv_path)


def make_epoch_metrics_row(
    epoch: int,
    train_log: Dict[str, float],
    val_log: Dict[str, float],
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    is_best: bool,
    checkpoint_path: str,
    selection_metric_name: str,
    selection_metric_value: float,
    selection_metric_source: str,
    best_metric_mode: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "epoch": int(epoch),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "is_best": int(bool(is_best)),
        "best_metric": to_plain_number(best_metric),
        "checkpoint_path": checkpoint_path,
        "selection_metric_name": selection_metric_name,
        "selection_metric_source": selection_metric_source,
        "selection_metric_value": to_plain_number(selection_metric_value),
        "best_metric_mode": best_metric_mode,
    }
    for k, v in train_log.items():
        row[f"train_{k}"] = to_plain_number(v)
    for k, v in val_log.items():
        row[f"val_{k}"] = to_plain_number(v)
    return row


def infer_best_metric_mode(metric_name: str, requested: str = "auto") -> str:
    if requested in {"min", "max"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported best_metric_mode: {requested}")
    name = metric_name.lower()
    if any(k in name for k in ["loss", "error", "mae", "mse", "rmse", "ce", "nll"]):
        return "min"
    return "max"


def is_better_metric(current: float, best: float, mode: str) -> bool:
    if not np.isfinite(current):
        return False
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError(f"Unsupported metric mode: {mode}")


def select_checkpoint_metric(
    train_log: Dict[str, float],
    val_log: Optional[Dict[str, float]],
    metric_name: str,
) -> Tuple[float, str]:
    source = "val" if val_log is not None else "train"
    log = val_log if val_log is not None else train_log
    if metric_name not in log:
        available = ", ".join(sorted(log.keys()))
        raise KeyError(
            f"Metric '{metric_name}' was not found in {source}_log. Available: {available}. "
            "Try --save_best_by loss_total or --save_best_by bcd_f1."
        )
    return float(log[metric_name]), source


# =============================================================================
# Train / validate
# =============================================================================


def set_learnable_registration_trainable(model: nn.Module, enabled: bool) -> None:
    m = model.module if hasattr(model, "module") else model
    for name in ["learnable_reg_head", "residual_flow_head"]:
        sub = getattr(m, name, None)
        if sub is None:
            continue
        sub.train(enabled)
        for p in sub.parameters():
            p.requires_grad = enabled


def set_dino_trainable(model: nn.Module, enabled: bool) -> None:
    m = model.module if hasattr(model, "module") else model
    enc = getattr(m, "feature_extractor", None)
    if enc is not None:
        enc = getattr(enc, "encoder", enc)
    dino = getattr(enc, "dino", None)
    if dino is None:
        return
    dino.train(enabled)
    for p in dino.parameters():
        p.requires_grad = enabled


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    module: Any,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    trainable_params: Optional[List[torch.nn.Parameter]] = None,
    epoch: int = 0,
    phase: str = "train",
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    meters: Dict[str, float] = {}
    steps = 0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            H0_t1_to_t2 = batch.get("H0_t1_to_t2") if args.use_provided_homography else None
            out = model(batch["t1"], batch["t2"], H0_t1_to_t2=H0_t1_to_t2, task=args.task)
            loss, loss_log = compute_losses(
                out=out,
                args=args,
                module=module,
                target_bcd=batch.get("bcd"),
                label_a=batch.get("label_a"),
                label_b=batch.get("label_b"),
            )
            if is_train:
                if not loss.requires_grad:
                    raise RuntimeError("Loss does not require gradients. Check model outputs and frozen branches.")
                loss.backward()
                if args.clip_grad > 0 and trainable_params is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.clip_grad)
                optimizer.step()

        # Metrics.
        if "BCD" in out and "bcd" in batch:
            bcd_overlap = get_overlap_mask(out, out["BCD"], args, preferred=["BCD_overlap_used", "BCD_overlap_core"])
            target_for_metric = prepare_change_target_for_loss(
                module, batch["bcd"], bcd_overlap, args.bcd_num_classes,
                args.bcd_ignore_index, args.unknown_index, out["BCD"].shape[-2:]
            )
            for k, v in compute_change_metrics(out["BCD"], target_for_metric, None, args.bcd_ignore_index, args.unknown_index, "bcd").items():
                meters[k] = meters.get(k, 0.0) + float(v)
        if "SCD_logits" in out and "bcd" in batch:
            scd_overlap = get_overlap_mask(out, out["SCD_logits"], args, preferred=["SCD_overlap_used"])
            target_for_metric = prepare_change_target_for_loss(
                module, batch["bcd"], scd_overlap, 2,
                args.bcd_ignore_index, args.unknown_index, out["SCD_logits"].shape[-2:]
            )
            for k, v in compute_change_metrics(out["SCD_logits"], target_for_metric, None, args.bcd_ignore_index, args.unknown_index, "scd").items():
                meters[k] = meters.get(k, 0.0) + float(v)

        # Diagnostic logs from model outputs.
        diag = {
            "used_auto_coarse_ratio": bool_list_ratio(out.get("used_auto_coarse", [])),
            "used_residual_refine_ratio": bool_list_ratio(out.get("used_residual_refine", [])),
            "used_fallback_to_coarse_ratio": bool_list_ratio(out.get("used_fallback_to_coarse", [])),
            "unknown_output_ratio": 1.0 if bool(out.get("used_unknown_output", False)) else 0.0,
            "inlier_ratio": tensor_mean_float(out.get("inlier_ratio", torch.tensor(0.0, device=device))),
            "provided_H0_ratio": 1.0 if bool(out.get("provided_H0", False)) else 0.0,
            "trusted_provided_H0_ratio": 1.0 if bool(out.get("trusted_provided_H0", False)) else 0.0,
        }
        if torch.is_tensor(out.get("learnable_affine_raw", None)):
            diag["learnable_affine_raw_abs_mean"] = float(out["learnable_affine_raw"].detach().abs().mean().cpu())
        for k, v in diag.items():
            meters[k] = meters.get(k, 0.0) + float(v)

        for k, v in loss_log.items():
            meters[k] = meters.get(k, 0.0) + float(v)
        steps += 1

        if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == len(loader)):
            matches = safe_sum_lengths(out.get("matches", []))
            inliers = safe_sum_lengths(out.get("inlier_matches", []))
            loss_parts = " ".join(f"{k}={v:.6f}" for k, v in loss_log.items())
            names = batch.get("name", ["N/A"])
            first_name = names[0] if isinstance(names, (list, tuple)) else str(names)
            print(
                f"[{phase}][Epoch {epoch:04d}][{step:04d}/{len(loader):04d}] "
                f"{loss_parts} matches={matches} inliers={inliers} "
                f"auto={diag['used_auto_coarse_ratio']:.3f} fallback={diag['used_fallback_to_coarse_ratio']:.3f} "
                f"trustedH={diag['trusted_provided_H0_ratio']:.3f} "
                f"sample={first_name}"
            )

    if steps == 0:
        return {"loss_total": float("inf")}
    return {k: v / steps for k, v in meters.items()}


def make_dataset(args: argparse.Namespace, data_root: Union[str, Path]) -> CCSCD5PairDataset:
    return CCSCD5PairDataset(
        data_root=data_root,
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
        require_homography=not args.allow_missing_homography,
    )


def resolve_external_val_root(args: argparse.Namespace) -> Tuple[str, str]:
    if args.val_root.strip():
        return args.val_root.strip(), "val_root"
    if args.test_root.strip():
        return args.test_root.strip(), "test_root"
    return "", ""


def make_dataloaders(args: argparse.Namespace):
    dataset = make_dataset(args, args.data_root)
    external_val_root, external_val_source = resolve_external_val_root(args)
    val_set: Optional[Any] = None
    val_source = "none"

    if external_val_root:
        if args.val_ratio > 0:
            print(f"[Warn] --{external_val_source} is set, so --val_ratio is ignored.")
        train_set = dataset
        val_set = make_dataset(args, external_val_root)
        val_source = f"{external_val_source}:{external_val_root}"
    elif args.val_ratio > 0:
        val_size = max(1, int(round(len(dataset) * args.val_ratio)))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            raise ValueError("val_ratio is too large; no training samples remain.")
        generator = torch.Generator().manual_seed(args.seed)
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
        val_source = f"random_split:{args.val_ratio}"
    else:
        train_set = dataset

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
        )
    return train_loader, val_loader, dataset, val_set, val_source


# =============================================================================
# Argument parser
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train stabilized differentiable-registration change detection model.")

    # Dataset.  优先级：val_root > test_root > val_ratio随机划分 > 没有验证集
    parser.add_argument("--data_root", type=str, default="DATA/SCSCD7_3/train(256_256)")
    parser.add_argument("--val_root", type=str, default="DATA/SCSCD7_3/train(256_256)") #datasets_total/SCSCD7_3/test(256_256)
    parser.add_argument("--test_root", type=str, default="")
    parser.add_argument("--bcd_dir", type=str, default="bcd")
    parser.add_argument("--curr_img_dir", type=str, default="curr_img")
    parser.add_argument("--curr_seg_dir", type=str, default="curr_seg")
    parser.add_argument("--prev_img_dir", type=str, default="prev_img")
    parser.add_argument("--prev_seg_dir", type=str, default="prev_seg")
    parser.add_argument("--bcd_label_mode", type=str, default="already_012_255", choices=["binary_0_255", "already_012_255", "rgb"])



# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    set_seed(args.seed)

    best_metric_mode = infer_best_metric_mode(args.save_best_by, args.best_metric_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] device = {device}")
    print(f"[Info] model_module = {args.model_module}")
    print(f"[Info] task = {args.task}")
    print(f"[Info] data_root = {args.data_root}")
    print(f"[Info] save_dir = {args.save_dir}")
    print(f"[Info] stable defaults: refine={args.refine_coarse_homography}, flow={args.use_learnable_residual_flow}, semantic_guidance={args.use_semantic_guidance_for_bcd}")
    print(f"[Info] BCD: num_classes={args.bcd_num_classes}, label_mode={args.bcd_label_mode}, unknown_index={args.unknown_index}")
    print(f"[Info] provided H: use={args.use_provided_homography}, trust={args.trust_provided_homography}, homography_dir={args.homography_dir}, homography_csv={args.homography_csv}")
    print(f"[Info] losses: lambda_bcd={args.lambda_bcd}, lambda_scd={args.lambda_scd}, lambda_seg={args.lambda_seg}, lambda_reg_feature={args.lambda_reg_feature}")
    if args.test_root:
        print("[Warn] test_root is used as validation during training; it is no longer an unbiased final test set.")

    train_loader, val_loader, full_dataset, val_dataset_or_subset, val_source = make_dataloaders(args)
    print(f"[Info] valid paired samples in train root = {len(full_dataset)}")
    if len(full_dataset.missing) > 0:
        print(f"[Warn] skipped train samples with missing files = {len(full_dataset.missing)}")
        print(f"[Warn] first skipped stems = {full_dataset.missing[:10]}")
    print(f"[Info] train samples used = {len(train_loader.dataset)}")
    print(f"[Info] train batches = {len(train_loader)}")
    if val_loader is not None:
        print(f"[Info] validation source = {val_source}")
        print(f"[Info] validation samples = {len(val_loader.dataset)}")
        print(f"[Info] validation batches = {len(val_loader)}")
        if hasattr(val_dataset_or_subset, "missing") and len(val_dataset_or_subset.missing) > 0:
            print(f"[Warn] skipped validation samples with missing files = {len(val_dataset_or_subset.missing)}")
    else:
        print("[Warn] No validation loader. best.pth will be selected from training metrics.")

    sample0 = full_dataset[0]
    print(f"[Debug] first sample = {sample0['name']}")
    print(f"[Debug] T1 shape = {tuple(sample0['t1'].shape)}, T2 shape = {tuple(sample0['t2'].shape)}")
    print(f"[Debug] BCD unique = {torch.unique(sample0['bcd']).tolist()}")
    print(f"[Debug] prev_seg unique = {torch.unique(sample0['label_a']).tolist()[:40]}")
    print(f"[Debug] curr_seg unique = {torch.unique(sample0['label_b']).tolist()[:40]}")
    if args.use_provided_homography and "H0_t1_to_t2" in sample0:
        print(f"[Debug] H0_t1_to_t2 =\n{sample0['H0_t1_to_t2']}")

    model, cfg, module = build_model(args, device)
    optimizer_params = [p for p in model.parameters()]
    grad_clip_params = optimizer_params
    initial_trainable = [p for p in model.parameters() if p.requires_grad]
    if not initial_trainable:
        raise RuntimeError("No trainable parameters.")
    # Put all parameters in the optimizer so branches can be unfrozen later.
    # Frozen parameters have no gradient and will not update until requires_grad=True.
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)

    print(f"[Info] initial trainable parameter tensors = {len(initial_trainable)}")
    print(f"[Info] freeze_dino = {args.freeze_dino}")
    print(f"[Info] save_best_by = {args.save_best_by} ({best_metric_mode})")

    best_metric = float("inf") if best_metric_mode == "min" else -float("inf")
    best_path = save_dir / "best.pth"
    last_path = save_dir / "last.pth"
    metrics_csv_path = resolve_metrics_csv_path(args)
    epoch_history = [] if args.reset_metrics_log else load_epoch_metrics_csv(metrics_csv_path)
    print(f"[Info] epoch metrics CSV = {metrics_csv_path}")

    for epoch in range(1, args.epochs + 1):
        reg_trainable = epoch > args.freeze_learnable_registration_epochs
        set_learnable_registration_trainable(model, reg_trainable)
        if args.unfreeze_dino_after_epoch >= 0:
            set_dino_trainable(model, epoch > args.unfreeze_dino_after_epoch)
        print(f"[Info] epoch {epoch}: learnable registration trainable = {reg_trainable}")

        train_log = run_one_epoch(
            model=model,
            loader=train_loader,
            args=args,
            module=module,
            device=device,
            optimizer=optimizer,
            trainable_params=grad_clip_params,
            epoch=epoch,
            phase="train",
        )

        val_log: Optional[Dict[str, float]] = None
        if val_loader is not None:
            val_log = run_one_epoch(
                model=model,
                loader=val_loader,
                args=args,
                module=module,
                device=device,
                optimizer=None,
                trainable_params=None,
                epoch=epoch,
                phase="val",
            )

        metric, metric_source = select_checkpoint_metric(train_log, val_log, args.save_best_by)
        is_best = is_better_metric(metric, best_metric, best_metric_mode)
        if is_best:
            best_metric = metric

        train_parts = " ".join(f"train_{k}={v:.6f}" for k, v in train_log.items())
        val_parts = ""
        if val_log is not None:
            val_parts = " " + " ".join(f"val_{k}={v:.6f}" for k, v in val_log.items())
        print(
            f"[Epoch {epoch:04d}/{args.epochs:04d}] {train_parts}{val_parts} "
            f"select_{metric_source}_{args.save_best_by}={metric:.6f} "
            f"best_{args.save_best_by}={best_metric:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_metric_name": args.save_best_by,
            "best_metric_mode": best_metric_mode,
            "last_metric": metric,
            "last_metric_source": metric_source,
            "train_log": train_log,
            "val_log": val_log,
            "args": vars(args),
            "cfg": asdict(cfg) if is_dataclass(cfg) else vars(cfg),
            "encoder_mode": "dino_only",
            "task": args.task,
        }
        checkpoint_path = str(last_path)
        torch.save(ckpt, last_path)
        if is_best:
            torch.save(ckpt, best_path)
            checkpoint_path = str(best_path)
            print(f"[Info] saved best checkpoint: {best_path} ({metric_source}_{args.save_best_by}={metric:.6f})")
        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_path = save_dir / f"epoch_{epoch:04d}.pth"
            torch.save(ckpt, epoch_path)

        epoch_row = make_epoch_metrics_row(
            epoch=epoch,
            train_log=train_log,
            val_log=val_log if val_log is not None else {},
            optimizer=optimizer,
            best_metric=best_metric,
            is_best=is_best,
            checkpoint_path=checkpoint_path,
            selection_metric_name=args.save_best_by,
            selection_metric_value=metric,
            selection_metric_source=metric_source,
            best_metric_mode=best_metric_mode,
        )
        epoch_history = [row for row in epoch_history if str(row.get("epoch", "")) != str(epoch)]
        epoch_history.append(epoch_row)
        save_epoch_metrics_csv(epoch_history, metrics_csv_path)
        print(f"[Metrics] saved epoch metrics to: {metrics_csv_path}")

    print("[Info] training finished")
    print(f"[Info] last checkpoint: {last_path}")
    print(f"[Info] best checkpoint: {best_path if best_path.exists() else 'not saved'}")
    print(f"[Info] best selection metric: {args.save_best_by}={best_metric} ({best_metric_mode})")


if __name__ == "__main__":
    main()
