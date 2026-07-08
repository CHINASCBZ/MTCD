import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from model.blocks.fpn import FPN, DsBnRelu
from model.blocks.cbam import CBAM
from model.blocks.adapter import DINOV3Wrapper, DenseAdapterLite
from model.blocks.diffatts import TransformerBlock
from model.blocks.refine import LearnableSoftMorph
from model.backbone.mobilenetv2 import mobilenet_v2


def get_backbone(backbone_name):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]        #手动设置了通道，有5个特征层，尺寸依次减少2倍
        # backbone.channels = [128, 128, 128, 128, 128]        #手动设置了通道，有5个特征层，尺寸依次减少2倍
    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


class PyramidFeatureFusion(nn.Module):
    def __init__(
        self,
        in_dims=[128, 128, 128, 128],
        dense_dim=1024,
        patch_size=16,
        hidden_dim=256,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.dense_dim = dense_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.c4 = nn.Sequential(
            DsBnRelu(in_dims[3] + hidden_dim, in_dims[3]), CBAM(in_dims[3], 8)
        )
        self.c3 = nn.Sequential(
            DsBnRelu(in_dims[2] + hidden_dim, in_dims[2]), CBAM(in_dims[2], 8)
        )
        self.c2 = nn.Sequential(
            DsBnRelu(in_dims[1] + hidden_dim, in_dims[1]), CBAM(in_dims[1], 8)
        )
        self.c1 = nn.Sequential(
            DsBnRelu(in_dims[0] + hidden_dim, in_dims[0]), CBAM(in_dims[0], 8)
        )

    def forward(self, feas, ds_feas):
        # process backbone (CNN) features
        x1, x2, x3, x4 = (
            feas  # [B, 128, 64, 64], [B, 128, 32, 32], [B, 128, 16, 16], [B, 128, 8, 8]
        )
        a1, a2, a3, a4 = (
            ds_feas  # [B, 256, 64, 64], [B, 256, 32, 32], [B, 256, 16, 16], [B, 256, 8, 8]
        )

        x4 = torch.cat([x4, a4], 1)
        x4 = self.c4(x4)

        x3 = torch.cat([x3, a3], 1)
        x3 = self.c3(x3)

        x2 = torch.cat([x2, a2], 1)
        x2 = self.c2(x2)

        x1 = torch.cat([x1, a1], 1)
        x1 = self.c1(x1)

        return x1, x2, x3, x4

class DINOOnlyEncoder(nn.Module):
    """
    只使用 DINOv3 dense features 的 Encoder。

    输出格式必须保持为:
        p2, p3, p4, p5

    每层通道数投影到 fpn_channels，供 PartialOverlapVoronoiEncoderCD 使用。
    """

    def __init__(
        self,
        fpn_channels=128,
        dino_weight="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        device="cuda",
        extract_ids=[5, 11, 17, 23],
        dino_out_dim=1024,
    ):
        super().__init__()

        self.fpn_channels = fpn_channels

        self.dino = DINOV3Wrapper(
            weights_path=dino_weight,
            device=device,
            extract_ids=extract_ids,
        )

        # 原 ChangeDINO 里 dense_out_dim = fpn_channels * 2
        dense_out_dim = fpn_channels * 2

        self.dense_adp = DenseAdapterLite(
            in_dim=dino_out_dim,
            out_dim=dense_out_dim,
            bottleneck=fpn_channels // 2,
        )

        # 把 DINO adapter 输出的 256 通道投影到 128 通道
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dense_out_dim, fpn_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, fpn_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(4)
        ])

    def forward(self, x):
        ds_fea = self.dino(x)
        ds_fea = self.dense_adp(ds_fea)

        if not isinstance(ds_fea, (tuple, list)) or len(ds_fea) != 4:
            raise ValueError(
                "DINOOnlyEncoder expects DenseAdapterLite to return 4 feature maps. "
                f"Got type={type(ds_fea)}, len={len(ds_fea) if isinstance(ds_fea, (tuple, list)) else 'N/A'}"
            )

        p2, p3, p4, p5 = [
            proj(f) for proj, f in zip(self.proj, ds_fea)
        ]

        return p2, p3, p4, p5

class Encoder(nn.Module):
    def __init__(
        self,
        backbone="mobilenetv2",
        fpn_channels=128,
        deform_groups=4,
        gamma_mode="SE",
        beta_mode="contextgatedconv",

        dino_weight="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        device="cuda",
        extract_ids=[5, 11, 17, 23],
        **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone)
        self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(
            weights_path=dino_weight, device=device, extract_ids=extract_ids
        )
        self.dense_adp = DenseAdapterLite(
            in_dim=1024, out_dim=dense_out_dim, bottleneck=fpn_channels // 2
        )
        self.pff = PyramidFeatureFusion(
            in_dims=[fpn_channels] * 4,
            dense_dim=1024,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )

    def forward(self, x):
        """
        x1: [B, 3, H, W]
        x2: [B, 3, H, W]
        return: [B, 1, H, W]
        """
        #X:8,3,512,512
        fea = self.backbone.forward(x)
        # fea是一个含有5个特征的list，每个特征的通道数backbone.channels = [16, 24, 32, 96, 320]可以设置
        # 尺寸依次减少2倍：256，128,64,32,16
        #8,16,256,256  8,24,128,128 8,32,64,64  8,96,32,32 8,320,16,16
        fea = self.fpn(fea[-4:])  # t1_p1, t1_p2, t1_p3, t1_p4

        ds_fea = self.dino(x)

        # process dense features
        ds_fea = self.dense_adp(ds_fea)

        fea = self.pff(fea, ds_fea)

        return fea


class FuseGated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * dim, dim, 1, bias=True), nn.Sigmoid())
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([x1, x2], dim=1))
        fused = x2 + g * x1
        return self.mix(fused)


class Detector(nn.Module):
    def __init__(
        self,
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        **kwargs,
    ):
        super().__init__()
        self.p5_to_p4 = FuseGated(fpn_channels)
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)

        self.tb5 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="CDA",
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=3,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[0])
            ]
        )
        self.tb4 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="CDA",
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=3,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[1])
            ]
        )
        self.tb3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="OCDA",
                    window_size=8,
                    overlap_ratio=0.5,
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=2,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[2])
            ]
        )
        self.tb2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="OCDA",
                    window_size=8,
                    overlap_ratio=0.5,
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=1,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[3])
            ]
        )
        self.p5_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p4_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p3_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p2_head = nn.Conv2d(fpn_channels, 2, 1)

    def forward(self, x1s, x2s, size=(256, 256)):       #这里默认输出是256,256
        ### Extract backbone features
        t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        t2_p2, t2_p3, t2_p4, t2_p5 = x2s

        diff_p2 = torch.abs(t1_p2 - t2_p2)
        diff_p3 = torch.abs(t1_p3 - t2_p3)
        diff_p4 = torch.abs(t1_p4 - t2_p4)
        diff_p5 = torch.abs(t1_p5 - t2_p5)

        fea_p5 = self.tb5(diff_p5)
        pred_p5 = self.p5_head(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, diff_p4)
        fea_p4 = self.tb4(fea_p4)
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, diff_p3)
        fea_p3 = self.tb3(fea_p3)
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, diff_p2)
        fea_p2 = self.tb2(fea_p2)
        pred_p2 = self.p2_head(fea_p2)

        pred_p2 = F.interpolate(
            pred_p2, size=size, mode="bilinear", align_corners=False
        )
        pred_p3 = F.interpolate(
            pred_p3, size=size, mode="bilinear", align_corners=False
        )
        pred_p4 = F.interpolate(
            pred_p4, size=size, mode="bilinear", align_corners=False
        )
        pred_p5 = F.interpolate(
            pred_p5, size=size, mode="bilinear", align_corners=False
        )

        return pred_p2, pred_p3, pred_p4, pred_p5


class ChangeModel(nn.Module):
    def __init__(
        self, backbone="mobilenetv2", fpn_channels=128, n_layers=[1, 1, 1, 1], **kwargs
    ):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.detector = Detector(fpn_channels=fpn_channels, n_layers=n_layers, **kwargs)
        self.refiner = LearnableSoftMorph(3, 5)

    @torch.inference_mode()
    def _forward(self, x1, x2):
        # for inference
        fea1 = self.encoder(x1)
        fea2 = self.encoder(x2)
        pred, _, _, _ = self.detector(fea1, fea2, x1.shape[-2:])
        pred = self.refiner(pred)
        return pred

    def forward(self, x1, x2):
        # for training
        ## change detection
        fea1 = self.encoder(x1)
        fea2 = self.encoder(x2)

        preds = self.detector(fea1, fea2)
        final_pred = self.refiner(preds[0])
        return final_pred, preds  # pred, pred_p2, pred_p3, pred_p4, pred_p5

if __name__ == '__main__':
    x1 = torch.randn(1, 3, 512, 512)
    x2 = torch.randn(1, 3, 512, 512)
    model = ChangeModel(backbone="mobilenetv2")
    y1 = model(x1, x2)
    print(y1.shape)