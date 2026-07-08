

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import torchvision.utils as vutils
import cv2
import numpy as np
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------可视化代码-----------------------------#
def vis_feature_map(feat, out_size=None):
    """
    feat: [B,C,H,W]
    输出: [B,1,H,W]，把通道压成可视化热力图
    """
    x = feat.detach().float()

    # 通道维度取平均绝对响应
    x = x.abs().mean(dim=1, keepdim=True)

    if out_size is not None:
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

    # 每张图单独归一化到 0~1
    b = x.shape[0]
    x_flat = x.view(b, -1)
    x_min = x_flat.min(dim=1)[0].view(b, 1, 1, 1)
    x_max = x_flat.max(dim=1)[0].view(b, 1, 1, 1)
    x = (x - x_min) / (x_max - x_min + 1e-6)

    return x


def save_debug_visuals(T1, T2, enc1, enc2, feats1, feats2, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)

    # 只保存 batch 第 0 张
    T1_0 = T1[0:1].detach().cpu()
    T2_0 = T2[0:1].detach().cpu()

    vutils.save_image(T1_0, f"{save_dir}/step_{step}_T1.png", normalize=True)
    vutils.save_image(T2_0, f"{save_dir}/step_{step}_T2.png", normalize=True)

    out_size = T1.shape[-2:]

    for i, f in enumerate(enc1):
        img = vis_feature_map(f[0:1], out_size=out_size).cpu()
        vutils.save_image(img, f"{save_dir}/step_{step}_enc1_p{i+2}.png")

    for i, f in enumerate(enc2):
        img = vis_feature_map(f[0:1], out_size=out_size).cpu()
        vutils.save_image(img, f"{save_dir}/step_{step}_enc2_p{i+2}.png")

    for i, f in enumerate(feats1):
        img = vis_feature_map(f[0:1], out_size=out_size).cpu()
        vutils.save_image(img, f"{save_dir}/step_{step}_feats1_d{i+2}.png")

    for i, f in enumerate(feats2):
        img = vis_feature_map(f[0:1], out_size=out_size).cpu()
        vutils.save_image(img, f"{save_dir}/step_{step}_feats2_d{i+2}.png")

#-----------------------可视化代码-----------------------------#


# =============================================================================
# Basic utilities
# =============================================================================
def _best_group_count(channels: int, max_groups: int = 32) -> int:
    """Choose a valid GroupNorm group count for small-batch training."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvBnAct(nn.Module):
    """
    Conv + Norm + SiLU.

    Stabilized version:
    - Original file used BatchNorm everywhere.
    - For remote-sensing CD, batch size is often 1-2, so GroupNorm is safer.
    - Keep the class name to avoid changing the rest of the model code.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, norm: str = "group"):
        super().__init__()
        p = k // 2
        norm = norm.lower()
        if norm in {"bn", "batch", "batchnorm"}:
            norm_layer: nn.Module = nn.BatchNorm2d(out_ch)
        elif norm in {"none", "identity"}:
            norm_layer = nn.Identity()
        else:
            norm_layer = nn.GroupNorm(_best_group_count(out_ch), out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            norm_layer,
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

def ensure_4d_rgb(x: torch.Tensor) -> torch.Tensor:
    """Accept [C,H,W] or [B,C,H,W], return [B,3,H,W]."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected 3D/4D tensor, got shape={tuple(x.shape)}")
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[1] != 3:
        raise ValueError(f"Expected RGB tensor with C=3, got C={x.shape[1]}")
    return x


def _to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """Convert one image tensor [3,H,W] or [1,3,H,W] to RGB uint8 HWC."""
    if x.dim() == 4:
        if x.shape[0] != 1:
            raise ValueError("_to_uint8_image expects a single image when input is 4D.")
        x = x[0]
    x = x.detach().float().cpu()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if x.shape[0] != 3:
        raise ValueError(f"Expected C=3 or C=1, got C={x.shape[0]}")

    # Robust normalization. Works for [0,1], [0,255], or normalized tensors.
    mn = float(x.min())
    mx = float(x.max())
    if mx <= 1.5 and mn >= -0.5:
        x = x.clamp(0, 1) * 255.0
    else:
        x = (x - x.min()) / (x.max() - x.min() + 1e-6) * 255.0
    img = x.byte().permute(1, 2, 0).numpy()
    return img


def image_to_gray_uint8(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.ndim == 2:
        return img_rgb.astype(np.uint8)
    return cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def tensor_to_gray_uint8(x: torch.Tensor) -> np.ndarray:
    return image_to_gray_uint8(_to_uint8_image(x))


def make_grid_points(h: int, w: int, num_points: int) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    side = int(np.ceil(np.sqrt(num_points)))
    xs = np.linspace(0, max(w - 1, 0), side, dtype=np.float32)
    ys = np.linspace(0, max(h - 1, 0), side, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    return pts[:num_points].astype(np.float32)


def default_scale_homography(src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)


# =============================================================================
# Automatic coarse registration: ORB + RANSAC
# =============================================================================



# =============================================================================
# Sampling, Voronoi, prototypes, residual matching
# =============================================================================

def harris_canny_sampling(
    image: torch.Tensor,
    num_points: int = 512,
    canny_low: int = 50,
    canny_high: int = 150,
    min_distance: int = 8,
) -> np.ndarray:
    gray = tensor_to_gray_uint8(image)
    h, w = gray.shape
    gray_f = np.float32(gray) / 255.0
    harris = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    harris = cv2.normalize(harris, None, 0, 1, cv2.NORM_MINMAX)
    edges = cv2.Canny(gray, canny_low, canny_high).astype(np.float32) / 255.0
    score = 0.75 * harris + 0.25 * edges

    order = np.argsort(score.reshape(-1))[::-1]
    occupied = np.zeros((h, w), dtype=np.uint8)
    selected: List[Tuple[float, float]] = []
    for idx in order:
        if len(selected) >= num_points:
            break
        y = int(idx // w)
        x = int(idx % w)
        if score[y, x] <= 0:
            break
        y0, y1 = max(0, y - min_distance), min(h, y + min_distance + 1)
        x0, x1 = max(0, x - min_distance), min(w, x + min_distance + 1)
        if occupied[y0:y1, x0:x1].any():
            continue
        selected.append((float(x), float(y)))
        occupied[y0:y1, x0:x1] = 1

    if len(selected) < max(8, num_points // 8):
        return make_grid_points(h, w, num_points)
    if len(selected) < num_points:
        grid = make_grid_points(h, w, num_points - len(selected))
        selected_np = np.array(selected, dtype=np.float32)
        return np.concatenate([selected_np, grid], axis=0)[:num_points]
    return np.array(selected, dtype=np.float32)


def voronoi_partition(image_hw: Tuple[int, int], seeds_xy: np.ndarray) -> np.ndarray:
    h, w = image_hw
    seeds_xy = np.asarray(seeds_xy, dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float32)
    tree = cKDTree(seeds_xy)
    _, labels = tree.query(pix, k=1)
    return labels.reshape(h, w).astype(np.int64)



def bidirectional_topk_match(sim: torch.Tensor, k: int = 5, min_sim: float = 0.30) -> List[Tuple[int, int, float]]:
    r1, r2 = sim.shape
    if r1 == 0 or r2 == 0:
        return []
    k1, k2 = min(k, r2), min(k, r1)
    top_j = torch.topk(sim, k=k1, dim=1).indices
    top_i = torch.topk(sim, k=k2, dim=0).indices
    top_i_sets = [set(top_i[:, j].detach().cpu().tolist()) for j in range(r2)]
    matches: List[Tuple[int, int, float]] = []
    for i in range(r1):
        for j in top_j[i].detach().cpu().tolist():
            score = float(sim[i, j].detach().cpu())
            if score < min_sim:
                continue
            if i in top_i_sets[j]:
                matches.append((i, j, score))
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches




# =============================================================================
# Warping and overlap
# =============================================================================

def _scale_pixel_coord(src_size: int, dst_size: int) -> float:
    """Map pixel coordinates [0, src_size-1] to [0, dst_size-1]."""
    if src_size > 1 and dst_size > 1:
        return float(dst_size - 1) / float(src_size - 1)
    return float(dst_size) / max(float(src_size), 1.0)


def _make_image_to_feature_scale_tensor(
    img_hw: Tuple[int, int],
    feat_hw: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    img_h, img_w = img_hw
    feat_h, feat_w = feat_hw
    sx = _scale_pixel_coord(img_w, feat_w)
    sy = _scale_pixel_coord(img_h, feat_h)
    return torch.tensor(
        [[sx, 0.0, 0.0],
         [0.0, sy, 0.0],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def image_homography_to_feature_homography_torch(
    H_img: torch.Tensor,
    src_img_hw: Tuple[int, int],
    dst_img_hw: Tuple[int, int],
    src_feat_hw: Tuple[int, int],
    dst_feat_hw: Tuple[int, int],
) -> torch.Tensor:
    """Differentiable image-coordinate -> feature-coordinate homography."""
    device = H_img.device
    dtype = H_img.dtype
    if H_img.dim() == 2:
        H_img = H_img.unsqueeze(0)
    S_dst = _make_image_to_feature_scale_tensor(dst_img_hw, dst_feat_hw, device, dtype)
    S_src = _make_image_to_feature_scale_tensor(src_img_hw, src_feat_hw, device, dtype)
    S_src_inv = torch.linalg.inv(S_src)
    return S_dst.unsqueeze(0) @ H_img @ S_src_inv.unsqueeze(0)


def warp_tensor_grid_sample_single_torchH(
    feat_single: torch.Tensor,
    H_src_to_dst: torch.Tensor,
    dst_hw: Tuple[int, int],
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Differentiable warp for one feature map with tensor homography.
    Gradients can flow to both feat_single and H_src_to_dst.
    """
    device = feat_single.device
    dtype = feat_single.dtype
    c, src_h, src_w = feat_single.shape
    dst_h, dst_w = dst_hw

    H = H_src_to_dst.to(device=device, dtype=torch.float32)
    H_inv = torch.linalg.inv(H)

    yy, xx = torch.meshgrid(
        torch.arange(dst_h, device=device, dtype=torch.float32),
        torch.arange(dst_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xx)
    dst_pts = torch.stack([xx, yy, ones], dim=0).reshape(3, -1)

    src_pts = H_inv @ dst_pts
    xs = src_pts[0] / (src_pts[2] + 1e-6)
    ys = src_pts[1] / (src_pts[2] + 1e-6)

    if src_w > 1:
        grid_x = 2.0 * xs / float(src_w - 1) - 1.0
    else:
        grid_x = torch.zeros_like(xs)
    if src_h > 1:
        grid_y = 2.0 * ys / float(src_h - 1) - 1.0
    else:
        grid_y = torch.zeros_like(ys)

    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, dst_h, dst_w, 2)
    warped = F.grid_sample(
        feat_single.unsqueeze(0).to(torch.float32),
        grid,
        mode=interpolation,
        padding_mode="zeros",
        align_corners=True,
    )
    return warped[0].to(dtype=dtype)


def warp_tensor_batch_grid_sample_torchH(
    feat: torch.Tensor,
    H_list: Union[torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor], np.ndarray],
    src_img_hw: Tuple[int, int],
    dst_img_hw: Tuple[int, int],
    target_feat_hw: Tuple[int, int],
    interpolation: str = "bilinear",
    normalize: bool = False,
) -> torch.Tensor:
    """Differentiable batch warp. Supports tensor H with gradient."""
    b, _, src_fh, src_fw = feat.shape
    dst_fh, dst_fw = target_feat_hw
    H_img = _homography_batch_to_tensor(H_list, device=feat.device, dtype=torch.float32)
    if H_img.shape[0] == 1 and b > 1:
        H_img = H_img.expand(b, -1, -1)
    if H_img.shape[0] != b:
        raise ValueError(f"H batch size {H_img.shape[0]} does not match feature batch size {b}")

    H_feat = image_homography_to_feature_homography_torch(
        H_img=H_img,
        src_img_hw=src_img_hw,
        dst_img_hw=dst_img_hw,
        src_feat_hw=(src_fh, src_fw),
        dst_feat_hw=(dst_fh, dst_fw),
    )

    warped = []
    for i in range(b):
        wi = warp_tensor_grid_sample_single_torchH(
            feat_single=feat[i],
            H_src_to_dst=H_feat[i],
            dst_hw=(dst_fh, dst_fw),
            interpolation=interpolation,
        )
        warped.append(wi.unsqueeze(0))
    out = torch.cat(warped, dim=0)
    if normalize:
        out = F.normalize(out, dim=1)
    return out

def warp_tensor_batch_grid_sample(
    feat: torch.Tensor,
    H_list: Union[torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor], np.ndarray],
    src_img_hw: Tuple[int, int],
    dst_img_hw: Tuple[int, int],
    target_feat_hw: Tuple[int, int],
    interpolation: str = "bilinear",
    normalize: bool = False,
) -> torch.Tensor:
    """
    Differentiable warp [B,C,Hf,Wf] from source image coordinates
    to destination feature coordinates.

    Compared with the original version, this function now accepts tensor homographies.
    If H_list is a tensor produced by a learnable registration head, gradients flow
    from the downstream loss back to that head through F.grid_sample.
    """
    # Tensor path: keep gradient to H.
    if torch.is_tensor(H_list):
        return warp_tensor_batch_grid_sample_torchH(
            feat=feat,
            H_list=H_list,
            src_img_hw=src_img_hw,
            dst_img_hw=dst_img_hw,
            target_feat_hw=target_feat_hw,
            interpolation=interpolation,
            normalize=normalize,
        )

    # Backward-compatible numpy/list path.
    b, _, src_fh, src_fw = feat.shape
    dst_fh, dst_fw = target_feat_hw

    warped_list = []

    for i in range(b):
        H_feat = image_homography_to_feature_homography(
            H_img=_to_numpy_homography(H_list[i]),
            src_img_hw=src_img_hw,
            dst_img_hw=dst_img_hw,
            src_feat_hw=(src_fh, src_fw),
            dst_feat_hw=(dst_fh, dst_fw),
        )

        wi = warp_tensor_grid_sample_single(
            feat_single=feat[i],
            H_src_to_dst=H_feat,
            dst_hw=(dst_fh, dst_fw),
            interpolation=interpolation,
        )

        warped_list.append(wi.unsqueeze(0))

    out = torch.cat(warped_list, dim=0)

    if normalize:
        out = F.normalize(out, dim=1)

    return out

def _to_numpy_homography(H: Union[np.ndarray, torch.Tensor, List[List[float]]]) -> np.ndarray:
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()
    H = np.asarray(H, dtype=np.float32)
    if H.shape == (1, 3, 3):
        H = H[0]
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be [3,3], got {H.shape}")
    return H.astype(np.float32)


def image_homography_to_feature_homography(
    H_img: np.ndarray,
    src_img_hw: Tuple[int, int],
    dst_img_hw: Tuple[int, int],
    src_feat_hw: Tuple[int, int],
    dst_feat_hw: Tuple[int, int],
) -> np.ndarray:
    src_h, src_w = src_img_hw
    dst_h, dst_w = dst_img_hw
    src_fh, src_fw = src_feat_hw
    dst_fh, dst_fw = dst_feat_hw
    sx_dst = _scale_pixel_coord(dst_w, dst_fw)
    sy_dst = _scale_pixel_coord(dst_h, dst_fh)
    sx_src = _scale_pixel_coord(src_w, src_fw)
    sy_src = _scale_pixel_coord(src_h, src_fh)
    S_dst = np.array([[sx_dst, 0, 0], [0, sy_dst, 0], [0, 0, 1]], dtype=np.float32)
    S_src_inv = np.array([[1.0 / max(sx_src, 1e-6), 0, 0], [0, 1.0 / max(sy_src, 1e-6), 0], [0, 0, 1]], dtype=np.float32)
    return (S_dst @ H_img @ S_src_inv).astype(np.float32)


def warp_tensor_cv2_single(
    feat_single: torch.Tensor,
    H_src_to_dst: np.ndarray,
    dst_hw: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> torch.Tensor:
    """Warp a single tensor [C,H,W] to [C,dst_h,dst_w] with OpenCV."""
    H = _to_numpy_homography(H_src_to_dst)
    arr = feat_single.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    dst_h, dst_w = dst_hw
    warped = cv2.warpPerspective(
        arr,
        H,
        (dst_w, dst_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if warped.ndim == 2:
        warped = warped[..., None]
    warped = np.transpose(warped, (2, 0, 1))
    return torch.from_numpy(warped).to(device=feat_single.device, dtype=feat_single.dtype)


def warp_tensor_batch_cv2(
    feat: torch.Tensor,
    H_list: Sequence[np.ndarray],
    src_img_hw: Tuple[int, int],
    dst_img_hw: Tuple[int, int],
    target_feat_hw: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
    normalize: bool = False,
) -> torch.Tensor:
    """Warp [B,C,Hf,Wf] from source image coordinates to dst feature coordinates."""
    b, _, src_fh, src_fw = feat.shape
    dst_fh, dst_fw = target_feat_hw
    warped_list = []
    for i in range(b):
        H_feat = image_homography_to_feature_homography(
            H_img=_to_numpy_homography(H_list[i]),
            src_img_hw=src_img_hw,
            dst_img_hw=dst_img_hw,
            src_feat_hw=(src_fh, src_fw),
            dst_feat_hw=(dst_fh, dst_fw),
        )
        wi = warp_tensor_cv2_single(feat[i], H_feat, (dst_fh, dst_fw), interpolation=interpolation)
        warped_list.append(wi.unsqueeze(0))
    out = torch.cat(warped_list, dim=0)
    if normalize:
        out = F.normalize(out, dim=1)
    return out


def warp_feature_pyramid(
    source_feats: Sequence[torch.Tensor],
    H_src_to_dst_list: Sequence[np.ndarray],
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
    target_feats: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    warped = []

    for f_src, f_dst in zip(source_feats, target_feats):
        dst_feat_hw = tuple(f_dst.shape[-2:])

        wf = warp_tensor_batch_grid_sample(
            feat=f_src,
            H_list=H_src_to_dst_list,
            src_img_hw=src_hw,
            dst_img_hw=dst_hw,
            target_feat_hw=dst_feat_hw,
            interpolation="bilinear",
            normalize=True,
        )

        warped.append(wf)

    return tuple(warped)


def compute_overlap_mask_batch(
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
    H_src_to_dst_list: Union[torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor], np.ndarray],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute overlap mask. The mask itself is not used as a differentiable signal,
    so tensor homographies are detached here for the cv2 mask rasterization.
    """
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    ones = np.ones((src_h, src_w), dtype=np.float32)
    masks = []

    if torch.is_tensor(H_src_to_dst_list):
        H_iter = H_src_to_dst_list.detach().cpu().numpy()
    elif isinstance(H_src_to_dst_list, np.ndarray) and H_src_to_dst_list.ndim == 3:
        H_iter = H_src_to_dst_list
    else:
        H_iter = H_src_to_dst_list

    for H in H_iter:
        warped = cv2.warpPerspective(ones, _to_numpy_homography(H), (dst_w, dst_h), flags=cv2.INTER_NEAREST)
        mask = (warped > 0.5).astype(np.float32)
        masks.append(torch.from_numpy(mask)[None, None])
    return torch.cat(masks, dim=0).to(device=device, dtype=dtype)

def binary_erode(mask: torch.Tensor, k: int = 9, threshold: float = 0.5) -> torch.Tensor:
    """
    mask: [B, 1, H, W] or [B, H, W]
    return: [B, 1, H, W]

    用于训练阶段：
    只保留 overlap 内部更可信的 core 区域，去掉边界不稳定区域。
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    if k <= 1:
        return (mask > threshold).float()

    assert k % 2 == 1, "overlap_erode_ks should be odd"

    mask = (mask > threshold).float()
    pad = k // 2

    # erosion(mask) = 1 - dilation(1 - mask)
    inv = 1.0 - mask

    # 图像外部视为 non-overlap，避免边界被错误保留
    inv = F.pad(inv, (pad, pad, pad, pad), mode="constant", value=1.0)

    eroded = 1.0 - F.max_pool2d(inv, kernel_size=k, stride=1, padding=0)

    return eroded.clamp(0.0, 1.0)


def binary_dilate(mask: torch.Tensor, k: int = 3, threshold: float = 0.5) -> torch.Tensor:
    """
    mask: [B, 1, H, W] or [B, H, W]
    return: [B, 1, H, W]

    用于推理阶段：
    稍微扩大 overlap，避免真实 overlap 区域被错误打成 unknown。
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    if k <= 1:
        return (mask > threshold).float()

    assert k % 2 == 1, "infer_overlap_dilate_ks should be odd"

    mask = (mask > threshold).float()
    pad = k // 2

    mask = F.pad(mask, (pad, pad, pad, pad), mode="constant", value=0.0)
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=0)

    return dilated.clamp(0.0, 1.0)



def select_overlap_mask_for_stage(
    overlap_raw: torch.Tensor,
    training: bool,
    erode_ks: int = 9,
    dilate_ks: int = 3,
) -> torch.Tensor:
    """
    Use a conservative overlap core during training and a slightly dilated overlap
    during inference. This fixes the train/infer mismatch where the old code always
    used eroded overlap, even at eval time.
    """
    if training:
        return binary_erode(overlap_raw.float(), k=erode_ks)
    return binary_dilate(overlap_raw.float(), k=dilate_ks)






# =============================================================================
# Encoder adapter and fallback encoder
# =============================================================================

class SimpleFallbackEncoder(nn.Module):
    """Fallback encoder for smoke tests. Replace with ChangeDINO/ResNet encoder in real training."""
    def __init__(self, fpn_channels: int = 128):
        super().__init__()
        self.stem = nn.Sequential(ConvBnAct(3, fpn_channels, 3), ConvBnAct(fpn_channels, fpn_channels, 3))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBnAct(fpn_channels, fpn_channels, 3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBnAct(fpn_channels, fpn_channels, 3))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBnAct(fpn_channels, fpn_channels, 3))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p2 = self.stem(x)
        p3 = self.down2(p2)
        p4 = self.down3(p3)
        p5 = self.down4(p4)
        return p2, p3, p4, p5


class UserEncoderAdapter(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        backbone: str = "mobilenetv2",
        fpn_channels: int = 128,
        normalize: bool = True,
        use_fallback_if_missing: bool = True,
        **encoder_kwargs,
    ):

        return feats


# =============================================================================
# Decoders and heads
# =============================================================================

class TopDownFPNDecoder(nn.Module):
    def __init__(self, fpn_channels: int = 128, hidden: int = 128):
        super().__init__()
        self.lateral_conv = nn.ModuleList([ConvBnAct(fpn_channels, hidden, 3) for _ in range(4)])
        self.fuse_conv = nn.ModuleList([ConvBnAct(hidden * 2, hidden, 3) for _ in range(3)])

    def forward(self, encoder_feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p2, p3, p4, p5 = encoder_feats
        l2 = self.lateral_conv[0](p2)
        l3 = self.lateral_conv[1](p3)
        l4 = self.lateral_conv[2](p4)
        l5 = self.lateral_conv[3](p5)

        d4 = F.interpolate(l5, size=l4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.fuse_conv[0](torch.cat([d4, l4], dim=1))
        d3 = F.interpolate(d4, size=l3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.fuse_conv[1](torch.cat([d3, l3], dim=1))
        d2 = F.interpolate(d3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.fuse_conv[2](torch.cat([d2, l2], dim=1))
        return d2, d3, d4, l5


class SegmentationDecoderHead(nn.Module):
    """Semantic decoder using the decoded p2 feature."""
    def __init__(self, in_channels: int, num_classes: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = mid_channels or in_channels
        self.head = nn.Sequential(
            ConvBnAct(in_channels, mid_channels, 3),
            ConvBnAct(mid_channels, mid_channels, 3),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, decoded_feats: Sequence[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        logits = self.head(decoded_feats[0])
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

#BCD解码器
class OverlapAwarePyramidDetector(nn.Module):
    def __init__(self, fpn_channels: int = 128,
                 hidden: int = 192,
                 num_classes: int = 3,
                 semantic_guidance_channels: int = 0,):
        super().__init__()
        self.num_classes = num_classes
        self.semantic_guidance_channels = semantic_guidance_channels
        in_ch = 3 * fpn_channels + 1+semantic_guidance_channels
        self.proj = nn.ModuleList([
            nn.Sequential(
                ConvBnAct(in_ch, hidden, 3),
                ConvBnAct(hidden, hidden, 3),
                ConvBnAct(hidden, hidden, 3),
            )
            for _ in range(4)
        ])
        self.fuse = nn.Sequential(
            ConvBnAct(hidden * 4, hidden * 2, 3),
            ConvBnAct(hidden * 2, hidden * 2, 3),
            ConvBnAct(hidden * 2, hidden, 3),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

    def forward(
            self,
            reference_feats: Sequence[torch.Tensor],
            aligned_source_feats: Sequence[torch.Tensor],
            overlap_img: torch.Tensor,
            out_size: Tuple[int, int],
            sem_change_img: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p2_size = reference_feats[0].shape[-2:]
        level_feats = []


        return logits


class SemanticChangeDecoderHead(nn.Module):
    """
    A trainable semantic-change head.

    The original SCD output was produced by argmax(seg_A_to_T2) != argmax(seg_B),
    which is useful as a diagnostic but is not a trainable change branch.
    This head predicts 2-class SCD logits from aligned decoded features.
    """
    def __init__(self, fpn_channels: int = 128, hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.detector = OverlapAwarePyramidDetector(
            fpn_channels=fpn_channels,
            hidden=hidden,
            num_classes=num_classes,
            semantic_guidance_channels=0,
        )

    def forward(
        self,
        reference_feats: Sequence[torch.Tensor],
        aligned_source_feats: Sequence[torch.Tensor],
        overlap_img: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        return self.detector(
            reference_feats=reference_feats,
            aligned_source_feats=aligned_source_feats,
            overlap_img=overlap_img,
            out_size=out_size,
            sem_change_img=None,
        )


# =============================================================================
# Config
# =============================================================================

@dataclass
class ComprehensiveCDConfig:
    backbone: str = "mobilenetv2"
    fpn_channels: int = 128
    decoder_hidden: int = 128
    detector_hidden: int = 192

    # Tasks/classes
    seg_num_classes: int = 7
    bcd_num_classes: int = 2
    scd_unknown_index: int = 255
    compute_scd_from_seg: bool = True
    use_trainable_scd_head: bool = True
    scd_head_hidden: int = 128

    # overlap 训练 / 推理控制
    overlap_erode_ks: int = 9       # 训练时腐蚀 overlap，只保留可信 core
    infer_overlap_dilate_ks: int = 3  # 推理时膨胀 overlap，减少误判 unknown

    # BCD label convention
    unknown_index: int = 2
    bcd_ignore_index: int = 255

    # Output coordinate
    output_coordinate: str = "T2"
    assume_registered: bool = False

    # Provided homography control
    # If True and H0_t1_to_t2 is passed to forward(), the model trusts it as the final
    # T1->T2 transform and skips traditional residual refinement and learnable registration.
    # This is recommended when your dataset construction knows the exact crop offset, because
    # the model overlap mask will then match the 0/1/2 BCD labels more closely.
    trust_provided_homography: bool = True

    # Coarse alignment
    auto_coarse_align: bool = True
    auto_coarse_mode: str = "affine"  # affine or homography
    coarse_max_features: int = 5000
    coarse_keep_ratio: float = 0.25
    coarse_ransac_thresh: float = 5.0
    coarse_min_matches: int = 12
    coarse_min_inliers: int = 8
    coarse_fail_action: str = "scale"  # scale, identity, unknown

    # Traditional residual refinement
    # Stabilized default: False. Turn it on only after confirming coarse registration is reliable.
    refine_coarse_homography: bool = False
    fallback_to_coarse: bool = True
    num_seeds: int = 512
    match_level: int = 1
    topk: int = 5
    min_sim: float = 0.30
    ransac_thresh: float = 8.0
    min_inliers: int = 8
    prototype_alpha: float = 0.05

    # Differentiable affine registration branch
    use_learnable_registration: bool = True
    learnable_reg_level: int = 3
    learnable_reg_hidden: int = 128
    learnable_reg_max_translation_frac: float = 0.12
    learnable_reg_max_rotation_deg: float = 10.0
    learnable_reg_max_log_scale: float = 0.12
    learnable_reg_max_shear: float = 0.06
    learnable_reg_use_traditional_init: bool = True

    # Local residual flow refinement
    # Stabilized default: False, because flow can explain away real changes.
    use_learnable_residual_flow: bool = False
    residual_flow_hidden: int = 96
    residual_flow_max_feature_px: float = 4.0
    residual_flow_use_semantic_gate: bool = True
    residual_flow_sem_change_threshold: float = 0.35

    # Registration auxiliary losses
    reg_consistency_use_semantic_gate: bool = True
    reg_consistency_sem_change_threshold: float = 0.35

    # Encoder fallback
    use_fallback_encoder_if_missing: bool = True

    # Semantic guidance for BCD
    # Stabilized default: False. Enable after seg/SCD branch is already useful.
    use_semantic_guidance_for_bcd: bool = False
    detach_semantic_guidance: bool = True
    semantic_guidance_source: str = "trainable_scd"  # trainable_scd or seg_disagreement


# =============================================================================
# Differentiable registration heads
# =============================================================================

class DifferentiableAffineRegistrationHead(nn.Module):
    """
    STN-style residual affine registration head.

    Stabilized fix:
    - First warp feat1 with H_base into the T2 feature frame.
    - Then predict a small residual affine H_delta from already coarsely aligned features.
    - Compose as H_final = H_delta @ H_base.
    """

        return H_final, reg


class ResidualFlowRefinementHead(nn.Module):
    """
    Local differentiable residual alignment in destination feature coordinates.

    The head predicts a destination-to-source residual flow over the already aligned
    p2 feature map. Flow is applied with grid_sample, so BCD/SCD losses can update it.
    """



def warp_feature_by_residual_flow(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp a destination-frame feature map by a residual flow.
    flow is [B,2,H,W] in feature-pixel units and gives source sampling offset.
    """

    return F.grid_sample(feat.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True).to(feat.dtype)


def residual_flow_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    if flow.numel() == 0:
        return flow.sum() * 0.0
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def feature_consistency_loss(
    src_aligned: torch.Tensor,
    ref: torch.Tensor,
    overlap: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    src_n = F.normalize(src_aligned, dim=1)
    ref_n = F.normalize(ref, dim=1)
    loss_map = 1.0 - (src_n * ref_n).sum(dim=1, keepdim=True)
    if overlap is not None:
        if overlap.shape[-2:] != loss_map.shape[-2:]:
            overlap = F.interpolate(overlap.float(), size=loss_map.shape[-2:], mode="nearest")
        mask = (overlap > 0.5).to(loss_map.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (loss_map * mask).sum() / denom
    return loss_map.mean()



def prepare_bcd_target_for_loss(
    target: torch.Tensor,
    overlap_mask: torch.Tensor,
    bcd_num_classes: int,
    ignore_index: int = 255,
    unknown_index: int = 2,
) -> torch.Tensor:
    """
    Utility for the training script.

    For 2-class BCD, non-overlap / unknown labels must be ignored.
    For 3-class BCD, non-overlap can be assigned to unknown_index.
    """
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]
    out = target.clone().long()
    if overlap_mask.dim() == 4:
        overlap_bool = overlap_mask[:, 0] > 0.5
    else:
        overlap_bool = overlap_mask > 0.5
    if overlap_bool.shape[-2:] != out.shape[-2:]:
        overlap_bool = F.interpolate(overlap_bool.float().unsqueeze(1), size=out.shape[-2:], mode="nearest")[:, 0] > 0.5
    if bcd_num_classes <= 2:
        out[out == unknown_index] = ignore_index
        out[~overlap_bool] = ignore_index
    else:
        out[~overlap_bool] = unknown_index
    return out


def invert_homography_for_output(
    H: Union[torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor], np.ndarray]
) -> Union[torch.Tensor, List[Optional[np.ndarray]]]:
    if torch.is_tensor(H):
        return torch.linalg.inv(H)
    out: List[Optional[np.ndarray]] = []
    for Hi in H:
        try:
            out.append(np.linalg.inv(_to_numpy_homography(Hi)).astype(np.float32))
        except np.linalg.LinAlgError:
            out.append(None)
    return out


def translation_homography_from_crop_offsets(
    start_y_t1: int,
    start_x_t1: int,
    start_y_t2: int,
    start_x_t2: int,
) -> np.ndarray:
    """
    Build the exact T1-crop -> T2-crop homography for pure crop-offset data.

    Coordinates in the saved crops are related by:
        x_global = x_t1_crop + start_x_t1 = x_t2_crop + start_x_t2
        y_global = y_t1_crop + start_y_t1 = y_t2_crop + start_y_t2

    Therefore:
        x_t2_crop = x_t1_crop + start_x_t1 - start_x_t2
        y_t2_crop = y_t1_crop + start_y_t1 - start_y_t2

    If shift_x = start_x_t2 - start_x_t1 and shift_y = start_y_t2 - start_y_t1,
    then H = [[1,0,-shift_x], [0,1,-shift_y], [0,0,1]].
    """
    dx = float(start_x_t1 - start_x_t2)
    dy = float(start_y_t1 - start_y_t2)
    return np.array(
        [[1.0, 0.0, dx],
         [0.0, 1.0, dy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

# =============================================================================
# Full model
# =============================================================================

class ComprehensiveUnregisteredMultiTaskCD(nn.Module):
    def __init__(
        self,
        cfg: ComprehensiveCDConfig = ComprehensiveCDConfig(),
        encoder: Optional[nn.Module] = None,
        **encoder_kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = UserEncoderAdapter(
            encoder=encoder,
            backbone=cfg.backbone,
            fpn_channels=cfg.fpn_channels,
            normalize=True,
            use_fallback_if_missing=cfg.use_fallback_encoder_if_missing,
            **encoder_kwargs,
        )
        self.fpn_decoder = TopDownFPNDecoder(fpn_channels=cfg.fpn_channels, hidden=cfg.decoder_hidden)
        self.seg_head = SegmentationDecoderHead(
            in_channels=cfg.decoder_hidden,
            num_classes=cfg.seg_num_classes,
            mid_channels=cfg.decoder_hidden,
        )
        semantic_guidance_channels = 1 if cfg.use_semantic_guidance_for_bcd else 0
        self.detector = OverlapAwarePyramidDetector(
            fpn_channels=cfg.decoder_hidden,
            hidden=cfg.detector_hidden,
            num_classes=cfg.bcd_num_classes,
            semantic_guidance_channels=semantic_guidance_channels,
        )

        if cfg.use_trainable_scd_head:
            self.scd_change_head = SemanticChangeDecoderHead(
                fpn_channels=cfg.decoder_hidden,
                hidden=cfg.scd_head_hidden,
                num_classes=2,
            )
        else:
            self.scd_change_head = None

        if cfg.use_learnable_registration:
            self.learnable_reg_head = DifferentiableAffineRegistrationHead(
                channels=cfg.decoder_hidden,
                hidden=cfg.learnable_reg_hidden,
                max_translation_frac=cfg.learnable_reg_max_translation_frac,
                max_rotation_deg=cfg.learnable_reg_max_rotation_deg,
                max_log_scale=cfg.learnable_reg_max_log_scale,
                max_shear=cfg.learnable_reg_max_shear,
            )
        else:
            self.learnable_reg_head = None

        if cfg.use_learnable_residual_flow:
            self.residual_flow_head = ResidualFlowRefinementHead(#残差语义流修正
                channels=cfg.decoder_hidden,
                hidden=cfg.residual_flow_hidden,
                max_feature_px=cfg.residual_flow_max_feature_px,
            )
        else:
            self.residual_flow_head = None

        self.debug_vis = True

    def _unknown_bcd_output(self, b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        logits = torch.zeros(b, self.cfg.bcd_num_classes, h, w, device=device, dtype=dtype)
        if self.cfg.bcd_num_classes >= 3:
            logits[:, 2] = 10.0
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        overlap = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
        return {"BCD": logits, "BCD_prob": prob, "BCD_pred": pred, "overlap_mask": overlap, "used_unknown_output": True}

    def _resolve_coarse_H_batch(
        self,
        T1: torch.Tensor,
        T2: torch.Tensor,
        h1: int,
        w1: int,
        h2: int,
        w2: int,
        H0_t1_to_t2: Optional[Union[np.ndarray, torch.Tensor, Sequence[Any]]],
    ) -> Tuple[Optional[List[np.ndarray]], List[Dict[str, Any]], List[bool]]:
        b = T1.shape[0]
        #分支一：
        #如果调用forward()时传入了H0_t1_to_t2参数，函数会直接使用这个矩阵，跳过所有自动配准步骤。
        if H0_t1_to_t2 is not None:
            if isinstance(H0_t1_to_t2, torch.Tensor) and H0_t1_to_t2.dim() == 3:
                H_list = [_to_numpy_homography(H0_t1_to_t2[i]) for i in range(b)]
            elif isinstance(H0_t1_to_t2, np.ndarray) and H0_t1_to_t2.ndim == 3:
                H_list = [_to_numpy_homography(H0_t1_to_t2[i]) for i in range(b)]
            elif isinstance(H0_t1_to_t2, (list, tuple)) and len(H0_t1_to_t2) == b:
                H_list = [_to_numpy_homography(H0_t1_to_t2[i]) for i in range(b)]
            else:
                H = _to_numpy_homography(H0_t1_to_t2)
                H_list = [H.copy() for _ in range(b)]
            return H_list, [{"ok": True, "method": "provided_H0"} for _ in range(b)], [False for _ in range(b)]

        #分支二：自动执行 ORB+RANSAC 粗配准
        #如果没有提供初始 H0 且cfg.auto_coarse_align=True，函数会逐样本执行 ORB 特征提取 + RANSAC 单应矩阵估计。
        H_list: List[np.ndarray] = []
        info_list: List[Dict[str, Any]] = []
        used_auto_list: List[bool] = []
        for i in range(b):
            H0 = None
            info: Dict[str, Any]
            used_auto = False
            # 步骤1：执行ORB+RANSAC自动粗配准
            if self.cfg.auto_coarse_align:
                H0, info = estimate_auto_coarse_homography_single(
                    T1[i],
                    T2[i],
                    mode=self.cfg.auto_coarse_mode,
                    max_features=self.cfg.coarse_max_features,
                    keep_ratio=self.cfg.coarse_keep_ratio,
                    ransac_thresh=self.cfg.coarse_ransac_thresh,
                    min_matches=self.cfg.coarse_min_matches,
                    min_inliers=self.cfg.coarse_min_inliers,
                )
                used_auto = H0 is not None
            else:
                info = {"ok": False, "reason": "auto_coarse_align_disabled"}

            #分支 3：配准失败后的降级策略
            """
            如果 ORB+RANSAC 配准失败（特征点太少、匹配太少或内点太少），
            函数会根据cfg.coarse_fail_action执行对应的降级策略，确保模型不会崩溃。
            """
            if H0 is None:
                action = self.cfg.coarse_fail_action.lower()
                if action == "unknown":
                    ## 策略1：返回全unknown输出（最严格）
                    return None, [info], [False]
                if action == "identity" and (h1, w1) == (h2, w2):
                    ## 策略2：使用单位矩阵（仅当T1和T2尺寸相同时）
                    H0 = np.eye(3, dtype=np.float32)
                    info = dict(info)
                    info.update({"fallback": "identity"})
                elif action in {"identity", "scale"}:
                    # # 策略3：使用缩放对齐矩阵（默认，最鲁棒）
                    H0 = default_scale_homography((h1, w1), (h2, w2))
                    info = dict(info)
                    info.update({"fallback": "scale" if action == "scale" else "scale_because_size_diff"})
                else:
                    raise ValueError(f"Unsupported coarse_fail_action={self.cfg.coarse_fail_action}")
                used_auto = False

            H_list.append(H0.astype(np.float32))
            info_list.append(info)
            used_auto_list.append(used_auto)
        return H_list, info_list, used_auto_list

    def _refine_H_batch(
        self,
        T2: torch.Tensor,
        feats1: Sequence[torch.Tensor],
        feats2: Sequence[torch.Tensor],
        h1: int,
        w1: int,
        h2: int,
        w2: int,
        H0_list: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[List[Tuple[int, int, float]]], List[List[Tuple[int, int, float]]], List[float], List[bool], List[bool]]:
        if not self.cfg.refine_coarse_homography:       #如果refine_coarse_homography为false，则就进入这个分支。
            #如果配置关闭精配准（refine_coarse_homography=False），型不会执行：不会做残差配准微调。
            #直接返回粗配准 H0，不做任何优化，所有精修相关变量填空 / 默认值
            b = T2.shape[0]
            return list(H0_list), [[] for _ in range(b)], [[] for _ in range(b)], [1.0 for _ in range(b)], [False for _ in range(b)], [False for _ in range(b)]


        #用深度特征做残差精修的核心步骤：
        #stage1：
        #把 T1 特征用 H0 先粗对齐到 T2 空间，用粗配准矩阵 H0 把 T1 的所有特征图变换到 T2 的坐标系
        feats1_coarse_to_t2 = warp_feature_pyramid(
            source_feats=feats1,
            H_src_to_dst_list=H0_list,
            src_hw=(h1, w1),
            dst_hw=(h2, w2),
            target_feats=feats2,
        )

        # stage2：
        level = int(self.cfg.match_level)
        level = max(0, min(level, len(feats2) - 1))


        final_H: List[np.ndarray] = []
        all_matches: List[List[Tuple[int, int, float]]] = []
        all_inliers: List[List[Tuple[int, int, float]]] = []
        ratios: List[float] = []
        used_residual: List[bool] = []
        used_fallback: List[bool] = []

        # stage3：
        # 对 batch 中每张图逐一精修（核心循环）
        for i in range(T2.shape[0]):
            H_res, matches, inliers, ratio = estimate_residual_homography_single(
                T2_single=T2[i],
                feat1_coarse_to_t2_single=feats1_coarse_to_t2[level][i],
                feat2_single=feats2[level][i],
                num_seeds=self.cfg.num_seeds,
                topk=self.cfg.topk,
                min_sim=self.cfg.min_sim,
                ransac_thresh=self.cfg.ransac_thresh,
                min_inliers=self.cfg.min_inliers,
                alpha=self.cfg.prototype_alpha,
            )
            if H_res is not None:


                feat_h, feat_w = feats2[level].shape[-2:]
                H_res_img = feature_residual_homography_to_image_homography(
                    H_res_feat=H_res,
                    feat_hw=(feat_h, feat_w),
                    img_hw=(h2, w2),
                )

                #最终矩阵 = 粗配准 × 残差精修（公式）
                H_final = (H_res_img @ H0_list[i]).astype(np.float32)

                final_H.append(H_final)
                used_residual.append(True)
                used_fallback.append(False)

            else:       ##H_res能够实现，执行下一步（精细化失败，参考点太少等问题）
                """
                如果精修失败（匹配点太少）
                退回使用粗配准 H0
                保证模型不崩溃
                """
                if self.cfg.fallback_to_coarse:
                    final_H.append(H0_list[i].copy())
                    used_residual.append(False)
                    used_fallback.append(True)
                else:
                    final_H.append(H0_list[i].copy())
                    used_residual.append(False)
                    used_fallback.append(False)
            all_matches.append(matches)
            all_inliers.append(inliers)
            ratios.append(float(ratio))
        return final_H, all_matches, all_inliers, ratios, used_residual, used_fallback

    #最终的分割头
    def _bcd_in_t2_with_H(
            self,
            feats1: Sequence[torch.Tensor],
            feats2: Sequence[torch.Tensor],
            h1: int,
            w1: int,
            h2: int,
            w2: int,
            H_t1_to_t2_list: Union[torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor], np.ndarray],
            sem_change_img: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:



        #（8）输出概率图 + 预测图
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        return {
            "BCD": logits,
            "BCD_prob": prob,
            "BCD_pred": pred,
            "overlap_mask": overlap_t2,
            "BCD_valid_mask_for_loss": overlap_used_t2,
            "feats1_warp_to_t2": feats1_warp_to_t2,
            "sem_change_for_bcd": sem_change_for_bcd,
            "BCD_overlap_raw": overlap_t2,
            "BCD_overlap_used": overlap_used_t2,
            "BCD_overlap_core": overlap_used_t2,
            "BCD_alignment_mask": alignment_mask,
            "residual_flow_p2": residual_flow_p2,
            "reg_feature_consistency_loss": reg_feat_loss,
            "reg_flow_smoothness_loss": reg_flow_smooth,
            "reg_flow_magnitude_loss": reg_flow_mag,
        }

    def _scd_outputs(
        self,
        decoded1: Sequence[torch.Tensor],
        decoded2: Sequence[torch.Tensor],
        h1: int,
        w1: int,
        h2: int,
        w2: int,
        H_t1_to_t2_list: Optional[Union[torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        seg_A = self.seg_head(decoded1, out_size=(h1, w1))      #语义分割头
        seg_B = self.seg_head(decoded2, out_size=(h2, w2))
        out: Dict[str, Any] = {"seg_A": seg_A, "seg_B": seg_B}

        if H_t1_to_t2_list is None:
            return out

        #把 T1 的语义分割结果 warp 到 T2 坐标系
        seg_A_to_T2 = warp_tensor_batch_grid_sample(
            feat=seg_A,
            H_list=H_t1_to_t2_list,
            src_img_hw=(h1, w1),
            dst_img_hw=(h2, w2),
            target_feat_hw=(h2, w2),
            interpolation="bilinear",
            normalize=False,
        )
        #计算 overlap 区域
        "这个函数就是把 T1 的有效图像区域通过配准矩阵投影到 T2，得到 T2 坐标系下的有效重叠区域 mask。"
        overlap_raw = compute_overlap_mask_batch(
            src_hw=(h1, w1),
            dst_hw=(h2, w2),
            H_src_to_dst_list=H_t1_to_t2_list,
            device=seg_B.device,
            dtype=seg_B.dtype,
        )
        #训练时通常会腐蚀 overlap，只保留更可靠的中心区域；推理时可能会稍微膨胀，避免误判太多 unknown 区域。
        overlap_used = select_overlap_mask_for_stage(
            overlap_raw.float(),
            training=self.training,
            erode_ks=self.cfg.overlap_erode_ks,
            dilate_ks=self.cfg.infer_overlap_dilate_ks,
        )

        out.update({
            "seg_A_to_T2": seg_A_to_T2,
            "SCD_overlap_mask": overlap_raw,
            "SCD_overlap_used": overlap_used,
        })

        # Diagnostic hard SCD from semantic argmax. Keep it, but do not rely on it as the only SCD branch.
        #生成一个基于语义类别差异的 SCD_from_seg
        if self.cfg.compute_scd_from_seg:
            #它不是通过可训练的 scd_change_head 预测出来的，而是用一个很直接的规则：
            """
            T1 对齐后的语义类别 != T2 的语义类别 → changed
            T1 对齐后的语义类别 == T2 的语义类别 → unchanged
            非重叠区域 → unknown
            """
            cls_A_t2 = torch.argmax(seg_A_to_T2, dim=1)
            cls_B = torch.argmax(seg_B, dim=1)
            scd_from_seg = (cls_A_t2 != cls_B).long()
            scd_from_seg = torch.where(
                overlap_raw[:, 0] > 0.5,
                scd_from_seg,
                torch.full_like(scd_from_seg, int(self.cfg.scd_unknown_index)),
            )
            out["SCD_from_seg"] = scd_from_seg

        # Trainable SCD head from aligned decoded features.
        #如果有可训练 SCD head，就执行主 SCD 分支（支持）
        if self.scd_change_head is not None:
            """
            如果模型里有可训练的 SCD 变化检测头，就用对齐后的 T1 特征和 T2 特征预测 SCD；
            如果没有可训练 SCD head，就退回使用前面由语义类别差异得到的 SCD_from_seg。
            """
            decoded1_to_t2 = warp_feature_pyramid(
                source_feats=decoded1,
                H_src_to_dst_list=H_t1_to_t2_list,
                src_hw=(h1, w1),
                dst_hw=(h2, w2),
                target_feats=decoded2,
            )
            scd_logits = self.scd_change_head(      #送入 SCD Head，进入class SemanticChangeDecoderHead(nn.Module)
                reference_feats=decoded2,
                aligned_source_feats=decoded1_to_t2,
                overlap_img=overlap_used,
                out_size=(h2, w2),
            )
            scd_prob = torch.softmax(scd_logits, dim=1)
            scd_pred = torch.argmax(scd_prob, dim=1)
            scd_pred = torch.where(
                overlap_used[:, 0] > 0.5,
                scd_pred,
                torch.full_like(scd_pred, int(self.cfg.scd_unknown_index)),
            )
            out.update({
                "SCD_logits": scd_logits,
                "SCD_prob": scd_prob,
                "SCD": scd_pred,
                "decoded1_to_T2_for_scd": decoded1_to_t2,
            })
        elif "SCD_from_seg" in out:
            out["SCD"] = out["SCD_from_seg"]

        return out



    def forward(
        self,
        T1: torch.Tensor,
        T2: torch.Tensor,
        H0_t1_to_t2: Optional[Union[np.ndarray, torch.Tensor, Sequence[Any]]] = None,
        task: str = "multi",
    ) -> Dict[str, Any]:
        """
        task:
          - "bcd": BCD only, requires/estimates registration.
          - "scd": semantic change, requires/estimates registration to produce SCD in T2.
          - "multi": seg_A/seg_B + SCD + BCD.
          - "seg": only seg_A/seg_B, no registration required.

        H0_t1_to_t2:
          Optional batch of image-coordinate homographies mapping T1 crop coordinates to
          T2/curr_img crop coordinates. For random crop offset data, pass the true matrix, e.g.
          [[1, 0, -shift_x], [0, 1, -shift_y], [0, 0, 1]].
          If cfg.trust_provided_homography=True, this matrix is used as the final transform.
        """
        task = task.lower()
        if task not in {"bcd", "scd", "multi", "seg"}:
            raise ValueError("task must be one of: 'bcd', 'scd', 'multi', 'seg'")

        T1 = ensure_4d_rgb(T1)
        T2 = ensure_4d_rgb(T2).to(T1.device)
        if T1.shape[0] != T2.shape[0]:
            raise ValueError(f"Batch sizes must match. Got T1 B={T1.shape[0]}, T2 B={T2.shape[0]}")
        b, _, h1, w1 = T1.shape
        _, _, h2, w2 = T2.shape
        device = T1.device

        if self.cfg.output_coordinate.upper() != "T2":
            raise NotImplementedError("This implementation currently supports output_coordinate='T2' only.")



        enc1 = self.feature_extractor(T1)
        enc2 = self.feature_extractor(T2)
        feats1 = self.fpn_decoder(enc1)
        feats2 = self.fpn_decoder(enc2)     #这个是有效果的，已经做了消融实验验证
        # feats1=enc1
        # feats2=enc2


        #------------------------------------可视化代码（训练时候关闭）--------------------------------------------------------#
        if getattr(self, "debug_vis", True):
            step = getattr(self, "global_step", 0)

            # if step % 100 == 0:
            if step == 1:
                save_debug_visuals(
                    T1=T1,
                    T2=T2,
                    enc1=enc1,
                    enc2=enc2,
                    feats1=feats1,
                    feats2=feats2,
                    save_dir="./debug_vis",
                    step=step,
                )

            self.global_step = step + 1

        # ------------------------------------可视化代码--------------------------------------------------------#


        outputs: Dict[str, Any] = {}
        need_alignment = task in {"bcd", "scd", "multi"}
        #"seg" 任务只做单时相语义分割，不需要 warp T1 → T2，所以 need_alignment=False。
        H0_list: Optional[List[np.ndarray]] = None
        #初始化 初步配准矩阵列表为空。
        H_final_list: Optional[Union[List[np.ndarray], torch.Tensor]] = None
        #初始化 最终配准矩阵列表为空
        provided_H0 = H0_t1_to_t2 is not None

        if need_alignment:
            if self.cfg.assume_registered:  #如果输入的已经配准好的模型（一般不执行，只有在对比实验中进行测试）
                if (h1, w1) != (h2, w2):    #如果 T1 和 T2 尺寸不同，则报错。
                    raise ValueError(f"assume_registered=True requires same size. Got T1={(h1, w1)}, T2={(h2, w2)}")
                H0_list = [np.eye(3, dtype=np.float32) for _ in range(b)]
                H_final_list = [np.eye(3, dtype=np.float32) for _ in range(b)]
                coarse_info = [{"ok": True, "method": "identity_registered"} for _ in range(b)]
                used_auto = [False for _ in range(b)]
                matches = [[] for _ in range(b)]
                inliers = [[] for _ in range(b)]
                ratios = [1.0 for _ in range(b)]
                used_residual = [False for _ in range(b)]
                used_fallback = [False for _ in range(b)]
            else:
                #_resolve_coarse_H_batch 会根据以下条件返回 batch 内每张图的初步配准矩阵 H0：
                #如果 H0_t1_to_t2 提供了初始矩阵，则直接使用。
                #如果没有提供，则尝试 自动粗配准（ORB + RANSAC）。


                H0_list, coarse_info, used_auto = self._resolve_coarse_H_batch(T1, T2, h1, w1, h2, w2, H0_t1_to_t2)

                ##一旦粗配准完全失败（H0_list = None），立刻停止后续所有流程，直接返回安全的默认输出，防止模型崩溃。
                if H0_list is None: #检查粗配准是否失败
                    #1. 生成并返回【全无效/全0】的变化检测输出（BCD）
                    outputs.update(self._unknown_bcd_output(b, h2, w2, device, feats2[0].dtype))
                    #2. 记录失败信息：粗配准信息 + 标记“使用了无效输出”
                    outputs.update({"coarse_info": coarse_info, "used_unknown_output": True})
                    # Still allow pure semantic outputs if task includes SCD.
                    ## 3. 如果任务是语义变化检测（SCD）或多任务，仍然允许输出语义结果
                    if task in {"scd", "multi"}:
                        outputs.update(self._scd_outputs(feats1, feats2, h1, w1, h2, w2, H_t1_to_t2_list=None))
                    return outputs

                # 如果用户提供了真实 H0_t1_to_t2，并且 cfg.trust_provided_homography=True，
                # 则直接把它作为最终配准矩阵。
                # 这样模型内部 overlap mask 会与数据构造中的 0/1/2 BCD 标签更一致。
                if provided_H0 and self.cfg.trust_provided_homography:
                    H_final_list = list(H0_list)
                    matches = [[] for _ in range(b)]
                    inliers = [[] for _ in range(b)]
                    ratios = [1.0 for _ in range(b)]
                    used_residual = [False for _ in range(b)]
                    used_fallback = [False for _ in range(b)]
                else:
                    # 如果粗配准成功:_refine_H_batch 对每张图执行 residual refinement：
                    #这个就是残差微调（核心步骤，粗配准 H0 → 用深度特征做残差精修 → 得到更准的最终 H_final）
                    H_final_list, matches, inliers, ratios, used_residual, used_fallback = self._refine_H_batch(
                        T2=T2,
                        feats1=feats1,  #使用深度特征进行残差微调
                        feats2=feats2,
                        h1=h1,
                        w1=w1,
                        h2=h2,
                        w2=w2,
                        H0_list=H0_list,
                    )

            ##断点
            # 可微配准分支：用一个可训练 affine head 预测残差矩阵，
            """
            如果开启了可学习配准头 + 不信任用户提供的 H0 →用深度学习网络（learnable_reg_head）对传统精配准结果做最后一次可微分微调，得到最终最准的 H_final。
            它是整个模型第二大创新点（传统精配准 + 深度学习可学习配准 双轨结构）。
            """
            learnable_reg_info: Dict[str, Any] = {}
            if self.learnable_reg_head is not None and not (provided_H0 and self.cfg.trust_provided_homography):
                #条件判断：只有满足 2 个条件才执行可学习配准
                #有可学习配准头（模型加载了深度学习配准网络）,不信任用户提供的 H0（必须自己优化，不能直接用外部给的矩阵）

                level_lr = int(self.cfg.learnable_reg_level)
                level_lr = max(0, min(level_lr, len(feats2) - 1))
                H_base_for_learnable = H_final_list if self.cfg.learnable_reg_use_traditional_init else [
                    default_scale_homography((h1, w1), (h2, w2)) for _ in range(b)
                ]
                H_final_list, learnable_reg_info = self.learnable_reg_head(
                    feat1=feats1[level_lr],
                    feat2=feats2[level_lr],
                    H_base=H_base_for_learnable,
                    src_hw=(h1, w1),
                    dst_hw=(h2, w2),
                )
                if "affine_param_l2" in learnable_reg_info:
                    outputs["reg_affine_param_l2_loss"] = learnable_reg_info["affine_param_l2"]
                outputs["learnable_H_delta"] = learnable_reg_info.get("H_delta")
                outputs["learnable_affine_raw"] = learnable_reg_info.get("raw_affine_params")

            #计算 T2 → T1 的逆单应矩阵
            H_t2_to_t1_list = invert_homography_for_output(H_final_list)

            outputs.update({
                "H0_t1_to_t2": H0_list,
                "H_t1_to_t2": H_final_list,
                "H_t2_to_t1": H_t2_to_t1_list,
                "coarse_info": coarse_info,
                "used_auto_coarse": used_auto,
                "matches": matches,
                "inlier_matches": inliers,
                "inlier_ratio": torch.tensor(ratios, device=device, dtype=torch.float32),
                "used_residual_refine": used_residual,
                "used_fallback_to_coarse": used_fallback,
                "used_unknown_output": False,
                "learnable_registration_enabled": self.learnable_reg_head is not None,
                "provided_H0": provided_H0,
                "trusted_provided_H0": bool(provided_H0 and self.cfg.trust_provided_homography),
            })


        """
        先做语义变化检测（SCD）→ 算出 “哪里语义发生了变化”→ 把这个信息喂给二值变化检测（BCD）→ 让 BCD 更准、更少误报！
        """
        sem_change_for_bcd: Optional[torch.Tensor] = None
        if task in {"seg", "scd", "multi"}:     #执行scd分支，如果任务包含 seg / scd / multi，就先执行语义分支
            outputs.update(self._scd_outputs(
                decoded1=feats1,
                decoded2=feats2,
                h1=h1,
                w1=w1,
                h2=h2,
                w2=w2,
                H_t1_to_t2_list=H_final_list if task in {"scd", "multi"} else None,
            ))

            #只有多任务（multi）+ 开启语义引导 → 才进入核心逻辑
            #这里使用了一个执行策略：
            if task == "multi" and self.cfg.use_semantic_guidance_for_bcd:
                #方式一：
                #方式 A：直接用训练好的 SCD 输出（最准、推荐）
                if self.cfg.semantic_guidance_source == "trainable_scd" and "SCD_prob" in outputs:
                    sem_change_for_bcd = outputs["SCD_prob"][:, 1:2] #直接取模型输出的 语义变化概率图
                elif "seg_A_to_T2" in outputs and "seg_B" in outputs:
                #方式B：用两个语义图自己算变化
                    pA = torch.softmax(outputs["seg_A_to_T2"], dim=1)
                    pB = torch.softmax(outputs["seg_B"], dim=1)
                    sem_same = torch.sum(pA * pB, dim=1, keepdim=True)
                    sem_change_for_bcd = 1.0 - sem_same
                    #计算 T1 语义 和 T2 语义的相似度，1 - 相似度 = 语义变化图
                else:
                    sem_change_for_bcd = None

                if sem_change_for_bcd is not None:
                    #阻断梯度（可选，让训练更稳）.detach() = 不让梯度从 BCD 回传给 SCD，两个任务互不干扰，训练更稳定
                    if self.cfg.detach_semantic_guidance:
                        sem_change_for_bcd = sem_change_for_bcd.detach()
                    outputs["sem_change"] = sem_change_for_bcd

        if task in {"bcd", "multi"}:    #
            if H_final_list is None:
                raise RuntimeError("Internal error: BCD requested but H_final_list is None.")

            #BCD任务分支
            outputs.update(self._bcd_in_t2_with_H(
                feats1=feats1,
                feats2=feats2,
                h1=h1,
                w1=w1,
                h2=h2,
                w2=w2,
                H_t1_to_t2_list=H_final_list,
                sem_change_img=sem_change_for_bcd,
            ))

        return outputs

def feature_residual_homography_to_image_homography(
    H_res_feat: np.ndarray,
    feat_hw: Tuple[int, int],
    img_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Convert residual homography from feature coordinates to image coordinates.

    H_res_feat:
        feature-coordinate homography, mapping T2 feature coords -> T2 feature coords.

    feat_hw:
        (feature_h, feature_w)

    img_hw:
        (image_h, image_w), here usually T2 image size.
    """
    H_res_feat = np.asarray(H_res_feat, dtype=np.float32)

    feat_h, feat_w = feat_hw
    img_h, img_w = img_hw

    sx = _scale_pixel_coord(img_w, feat_w)
    sy = _scale_pixel_coord(img_h, feat_h)
    S_img_to_feat = np.array(
        [
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    S_feat_to_img = np.linalg.inv(S_img_to_feat).astype(np.float32)

    H_res_img = S_feat_to_img @ H_res_feat @ S_img_to_feat

    return H_res_img.astype(np.float32)

# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from model.ChangeDINO import DINOOnlyEncoder

    cfg = ComprehensiveCDConfig(
        fpn_channels=128,
        decoder_hidden=128,
        detector_hidden=192,
        seg_num_classes=7,
        bcd_num_classes=2,

        auto_coarse_align=True,
        auto_coarse_mode="affine",
        coarse_fail_action="scale",

        refine_coarse_homography=False,
        num_seeds=128,
        match_level=1,
        topk=3,
        min_sim=0.20,
        min_inliers=4,

        use_fallback_encoder_if_missing=False,
        use_learnable_residual_flow=False,
        use_semantic_guidance_for_bcd=False,
    )

    encoder = DINOOnlyEncoder(
        fpn_channels=cfg.fpn_channels,
        dino_weight="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        device=device,
        extract_ids=[5, 11, 17, 23],
    ).to(device)

    model = ComprehensiveUnregisteredMultiTaskCD(       #main model
        cfg=cfg,
        encoder=encoder,
    ).to(device)

    model.eval()

    x1 = torch.rand(2, 3, 512, 256).to(device)
    x2 = torch.rand(2, 3, 256, 512).to(device)

    with torch.no_grad():
        out = model(x1, x2, task="multi")

    print("Available keys:", sorted(out.keys()))

    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape))
        elif isinstance(v, list):
            print(k, f"list(len={len(v)})")
        else:
            print(k, type(v))
