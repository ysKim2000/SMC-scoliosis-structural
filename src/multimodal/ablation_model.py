import torch
import torch.nn as nn
from timm import create_model

from model import (
    FiLMFusion, GatedFusion, SimpleConcatFusion, ProjConcatFusion, SelfAttentionFusion,
)


class BackboneModel(nn.Module):
    def __init__(self, model_type="resnet101.tv_in1k", pretrained=True, pool=True):
        super().__init__()
        self.feature_extractor = create_model(
            model_type,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg" if pool else "",
        )
        self.feature_dim = self.feature_extractor.num_features

    def forward(self, x):
        return self.feature_extractor(x)


class LumbarClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_p=0.4, out_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class ScoliosisLumbarAblationModel(nn.Module):
    """
    Method A:
    - Without clinical input, use the image-only head (no FiLM)
    - With clinical input, modulate image features via FiLM
    - clinical_only uses a separate clinical head
    """

    def __init__(
        self,
        pa_backbone="resnet101.tv_in1k",
        lat_backbone="resnet101.tv_in1k",
        clinical_dim=4,
        hidden_dim=256,
        dropout_p=0.4,
        use_pa=True,
        use_lat=True,
        use_clinical=True,
        fusion_type="film",
    ):
        super().__init__()
        self.use_pa = bool(use_pa)
        self.use_lat = bool(use_lat)
        self.use_clinical = bool(use_clinical)
        self.fusion_type = fusion_type

        if not (self.use_pa or self.use_lat or self.use_clinical):
            raise ValueError("At least one modality must be enabled.")

        # spatial (unpooled) features are only needed for self_attn fusion, and only
        # when there is actually an image+clinical fusion to perform
        will_fuse = (self.use_pa or self.use_lat) and self.use_clinical
        self.requires_spatial = will_fuse and fusion_type in ("self_attn", "cross_attn")
        pool_flag = not self.requires_spatial

        if self.use_pa:
            self.pa_model = BackboneModel(pa_backbone, pretrained=True, pool=pool_flag)
            pa_dim = self.pa_model.feature_dim
        else:
            pa_dim = 0

        if self.use_lat:
            self.lat_model = BackboneModel(lat_backbone, pretrained=True, pool=pool_flag)
            lat_dim = self.lat_model.feature_dim
        else:
            lat_dim = 0

        image_dim = pa_dim + lat_dim
        self.has_image = image_dim > 0

        if self.has_image and not self.requires_spatial:
            self.image_norm = nn.LayerNorm(image_dim)

        if self.has_image and self.use_clinical:
            self.fusion = self._build_fusion(fusion_type, image_dim, clinical_dim)
            classifier_in_dim = self.fusion.out_dim
        elif self.has_image:
            self.fusion = None
            classifier_in_dim = image_dim
        else:
            # clinical_only
            self.clinical_proj = nn.Sequential(
                nn.LayerNorm(clinical_dim),
                nn.Linear(clinical_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
            )
            classifier_in_dim = hidden_dim

        self.classifier = LumbarClassifier(
            input_dim=classifier_in_dim,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
            out_dim=1,
        )

    def _build_fusion(self, fusion_type, image_dim, clinical_dim):
        if fusion_type == "film":
            return FiLMFusion(image_dim, clinical_dim)
        elif fusion_type == "gated":
            return GatedFusion(image_dim, clinical_dim)
        elif fusion_type == "concat":
            return SimpleConcatFusion(image_dim, clinical_dim)
        elif fusion_type == "projection":
            return ProjConcatFusion(image_dim, clinical_dim, proj_dim=64)
        elif fusion_type == "self_attn":
            return SelfAttentionFusion(image_dim, clinical_dim, num_heads=8, num_layers=2)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, pa_data, lat_data, clinical):
        feats = []

        if self.use_pa:
            feats.append(self.pa_model(pa_data))
        if self.use_lat:
            feats.append(self.lat_model(lat_data))

        if self.has_image:
            img_feat = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
            if not self.requires_spatial:
                img_feat = self.image_norm(img_feat)
            fused = self.fusion(img_feat, clinical) if (self.fusion is not None) else img_feat
        else:
            fused = self.clinical_proj(clinical)

        out = self.classifier(fused)
        return out.squeeze(1)
