import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


# =========================================================
# 1. Backbone
# =========================================================
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ViT / DINO model prefixes that output (B, C) only — no spatial feature map
_VIT_PREFIXES = ('vit_', 'deit_', 'swin_', 'beit_', 'eva_')


def _parse_model_spec(spec):
    """Parse 'source:model_name[:weight_path]' or bare 'model_name' (→ timm).

    Supported sources:
      timm        : timm pretrained models (default)
      txv         : TorchXRayVision DenseNet  e.g. txv:densenet121-res224-all
      radimagenet : timm arch + RadImageNet weights
                    e.g. radimagenet:resnet50:/path/to/weights.pth
    """
    if ':' not in spec:
        return 'timm', spec, None
    parts = spec.split(':', 2)
    source     = parts[0].lower()
    model_name = parts[1]
    weight_path = parts[2] if len(parts) > 2 else None
    return source, model_name, weight_path


class BackboneModel(nn.Module):
    """Unified backbone wrapper.

    model_spec examples
    -------------------
    'resnet101.tv_in1k'                           → timm ImageNet pretrain
    'timm:resnet101.tv_in1k'                      → same
    'dinov2:vits14'                               → DINOv2 ViT-S/14 via torch.hub (384-dim)
    'dinov2:vitb14'                               → DINOv2 ViT-B/14 via torch.hub (768-dim)
    'txv:densenet121-res224-all'                  → TorchXRayVision multi-dataset DenseNet
    'radimagenet:resnet50:/path/weights.pth'      → RadImageNet weights on ResNet-50
    """

    def __init__(self, model_spec='resnet101.tv_in1k', pool=True):
        super().__init__()
        self.pool = pool

        source, model_name, weight_path = _parse_model_spec(model_spec)
        self.source          = source
        self.model_name      = model_name
        self._vit_input_size = None   # set below for ViT models

        # ── timm / RadImageNet ──────────────────────────────────────────────
        if source in ('timm', 'radimagenet'):
            pool_type = 'avg' if pool else ''

            # ViT/DINO patch_size=14 → input must be multiple of 14.
            # We keep our pipeline at 512 and resize inside forward.
            self._vit_input_size = None
            is_vit_model = model_name.lower().startswith(_VIT_PREFIXES)
            extra_kwargs = {}
            if is_vit_model:
                # nearest multiple of 14 below 512 → 504
                patch_size = 14
                native = (512 // patch_size) * patch_size  # 504
                self._vit_input_size = native
                extra_kwargs = dict(img_size=native, dynamic_img_size=True)

            self.feature_extractor = create_model(
                model_name,
                pretrained=(source == 'timm'),
                num_classes=0,
                global_pool=pool_type,
                **extra_kwargs,
            )
            if source == 'radimagenet':
                if weight_path is None:
                    raise ValueError("radimagenet source requires weight_path: 'radimagenet:arch:/path.pth'")
                state = torch.load(weight_path, map_location='cpu', weights_only=False)
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                # strip 'backbone.' prefix (RadImageNet checkpoint convention)
                if any(k.startswith('backbone.') for k in state):
                    state = {k[len('backbone.'):]: v for k, v in state.items()}
                # Remap Sequential-index keys → timm named keys
                # ResNet:    0→conv1, 1→bn1, 4→layer1, 5→layer2, 6→layer3, 7→layer4
                # DenseNet:  0.X → features.X
                _is_resnet = 'resnet' in model_name.lower()
                _RESNET_IDX = {'0': 'conv1', '1': 'bn1', '4': 'layer1',
                               '5': 'layer2', '6': 'layer3', '7': 'layer4'}
                def _remap_key(k):
                    parts = k.split('.')
                    if _is_resnet and parts[0] in _RESNET_IDX:
                        parts[0] = _RESNET_IDX[parts[0]]
                        return '.'.join(parts)
                    if not _is_resnet and parts[0] == '0' and len(parts) > 1:
                        return 'features.' + '.'.join(parts[1:])
                    return k
                state = {_remap_key(k): v for k, v in state.items()}
                missing, unexpected = self.feature_extractor.load_state_dict(state, strict=False)
                print(f"[RadImageNet] Loaded {weight_path} | missing={len(missing)} unexpected={len(unexpected)}")
            self.feature_dim = self.feature_extractor.num_features

        # ── DINOv2 (torch.hub) ──────────────────────────────────────────────
        elif source == 'dinov2':
            # model_name: vits14 | vitb14 | vitl14 | vitg14
            _DIM = {'vits14': 384, 'vitb14': 768, 'vitl14': 1024, 'vitg14': 1536}
            if model_name not in _DIM:
                raise ValueError(f"DINOv2 model_name must be one of {list(_DIM)}. Got '{model_name}'")
            self._dinov2 = torch.hub.load(
                'facebookresearch/dinov2', f'dinov2_{model_name}', verbose=False
            )
            self.feature_dim     = _DIM[model_name]
            self._vit_input_size = 518  # 14*37, DINOv2 native size

        # ── TorchXRayVision ─────────────────────────────────────────────────
        elif source == 'txv':
            import torchxrayvision as xrv
            # Silence the resolution warning for our 512×512 pipeline
            import warnings
            self._txv = xrv.models.DenseNet(weights=model_name)
            self._txv.op_threshs = None
            self.feature_dim = 1024  # DenseNet-121 feature dim
            # Buffers for undoing ImageNet normalization
            self.register_buffer('_mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('_std',  torch.tensor(_IMAGENET_STD ).view(1, 3, 1, 1))

        else:
            raise ValueError(f"Unknown backbone source '{source}'. Use: timm | dinov2 | txv | radimagenet")

    # ── is this a ViT-type model? (no spatial feature map) ──────────────────
    @property
    def is_vit(self):
        return self.model_name.lower().startswith(_VIT_PREFIXES)

    def forward(self, x):
        if self.source == 'dinov2':
            s = self._vit_input_size
            x = F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False)
            return self._dinov2(x)  # (B, embed_dim) CLS token

        if self.source == 'txv':
            # Undo ImageNet normalisation → grayscale → TXV [-1024, 1024] range
            x_raw  = x * self._std + self._mean          # (B,3,H,W)  [0,1]
            x_gray = x_raw.mean(dim=1, keepdim=True)     # (B,1,H,W)  [0,1]
            x_txv  = (2.0 * x_gray - 1.0) * 1024.0      # (B,1,H,W)  [-1024,1024]

            if self.pool:
                # features2 = DenseNet layers + /1024 norm + ReLU + GlobalAvgPool → (B,1024)
                return self._txv.features2(x_txv)
            else:
                # Spatial feature map (B,1024,H',W') for attention fusions
                feat = self._txv.features(x_txv / 1024.0)
                return F.relu(feat, inplace=True)

        if self._vit_input_size is not None:
            s = self._vit_input_size
            x = F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False)
        return self.feature_extractor(x)


# =========================================================
# 2. Fusion Modules
# =========================================================
class SimpleConcatFusion(nn.Module):
    def __init__(self, img_dim, clin_dim):
        super().__init__()
        self.out_dim = img_dim + clin_dim

    def forward(self, img_feat, clin_feat):
        return torch.cat([img_feat, clin_feat], dim=1)


class ProjConcatFusion(nn.Module):
    def __init__(self, img_dim, clin_dim, proj_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clin_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.out_dim = img_dim + proj_dim

    def forward(self, img_feat, clin_feat):
        clin_emb = self.proj(clin_feat)
        return torch.cat([img_feat, clin_emb], dim=1)


class FiLMFusion(nn.Module):
    def __init__(self, img_dim, clin_dim):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(clin_dim, img_dim * 2)
        )
        self.out_dim = img_dim

    def forward(self, img_feat, clin_feat):
        gamma_beta = self.film_gen(clin_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return img_feat * (1 + gamma) + beta


class PlaneAwareFiLM(nn.Module):
    """
    Radiographic-plane-routed FiLM: standing Cobb angle (coronal-plane measurement,
    matches the PA view) modulates PA features only; L1-S1 lordosis (sagittal-plane
    measurement, matches the LAT view) modulates LAT features only. Sex/Age are
    demographic, not plane-specific, so they are shared across both modulations.

    Assumes the "lumbar_only" clinical column order: [Sex, Age, Lumbar_Cobb, L1_S1_Lordosis].
    """
    def __init__(self, pa_dim, lat_dim, clin_dim,
                 sex_idx=0, age_idx=1, cobb_idx=2, lordosis_idx=3):
        super().__init__()
        assert clin_dim == 4, (
            "PlaneAwareFiLM assumes the lumbar_only clinical set "
            "[Sex, Age, Lumbar_Cobb, L1_S1_Lordosis] (clin_dim=4)."
        )
        self.shared_idx = [sex_idx, age_idx]
        self.cobb_idx = cobb_idx
        self.lordosis_idx = lordosis_idx

        self.pa_norm = nn.LayerNorm(pa_dim)
        self.lat_norm = nn.LayerNorm(lat_dim)
        self.film_gen_pa = nn.Linear(1 + len(self.shared_idx), pa_dim * 2)
        self.film_gen_lat = nn.Linear(1 + len(self.shared_idx), lat_dim * 2)
        self.out_dim = pa_dim + lat_dim

    def forward(self, pa_feat, lat_feat, clin_feat):
        shared = clin_feat[:, self.shared_idx]
        cobb = clin_feat[:, [self.cobb_idx]]
        lordosis = clin_feat[:, [self.lordosis_idx]]

        gamma_pa, beta_pa = torch.chunk(self.film_gen_pa(torch.cat([cobb, shared], dim=1)), 2, dim=1)
        gamma_lat, beta_lat = torch.chunk(self.film_gen_lat(torch.cat([lordosis, shared], dim=1)), 2, dim=1)

        pa_out = self.pa_norm(pa_feat) * (1 + gamma_pa) + beta_pa
        lat_out = self.lat_norm(lat_feat) * (1 + gamma_lat) + beta_lat
        return torch.cat([pa_out, lat_out], dim=1)


class GatedFusion(nn.Module):
    def __init__(self, img_dim, clin_dim):
        super().__init__()
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_dim, img_dim),
            nn.LayerNorm(img_dim)
        )
        self.gate = nn.Linear(img_dim * 2, img_dim)
        self.out_dim = img_dim

    def forward(self, img_feat, clin_feat):
        clin_emb = self.clin_proj(clin_feat)
        gate = torch.sigmoid(self.gate(torch.cat([img_feat, clin_emb], dim=1)))
        return gate * img_feat + (1 - gate) * clin_emb


class CrossAttentionFusion(nn.Module):
    def __init__(self, img_dim, clin_dim, num_heads=8):
        super().__init__()
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_dim, img_dim),
            nn.LayerNorm(img_dim)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(img_dim)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.out_dim = img_dim

    def forward(self, img_spatial, clin_feat):
        # img_spatial: (B, C, H, W)
        B, C, H, W = img_spatial.shape
        img_spatial = self.pool(img_spatial)                 # (B, C, 7, 7)
        img_tokens = img_spatial.view(B, C, -1).permute(0, 2, 1)  # (B, 49, C)

        clin_token = self.clin_proj(clin_feat).unsqueeze(1)       # (B, 1, C)

        attn_out, _ = self.cross_attn(
            query=clin_token,
            key=img_tokens,
            value=img_tokens
        )
        fused_token = self.layer_norm(clin_token + attn_out).squeeze(1)  # (B, C)
        return fused_token


class SelfAttentionFusion(nn.Module):
    def __init__(self, img_dim, clin_dim, num_heads=8, num_layers=2):
        super().__init__()
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_dim, img_dim),
            nn.LayerNorm(img_dim)
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        # 49 image tokens + 1 clinical token = 50
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, img_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=img_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(img_dim)
        self.out_dim = img_dim

    def forward(self, img_spatial, clin_feat):
        # img_spatial: (B, C, H, W)
        B, C, H, W = img_spatial.shape
        img_spatial = self.pool(img_spatial)                  # (B, C, 7, 7)

        img_tokens = img_spatial.view(B, C, -1).permute(0, 2, 1)  # (B, 49, C)
        clin_token = self.clin_proj(clin_feat).unsqueeze(1)       # (B, 1, C)

        tokens = torch.cat([clin_token, img_tokens], dim=1)       # (B, 50, C)
        tokens = tokens + self.pos_embed

        updated_tokens = self.transformer_encoder(tokens)
        final_token = self.layer_norm(updated_tokens[:, 0, :])     # clinical token
        return final_token


# =========================================================
# 3. Classifier
# =========================================================
class LumbarClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_p=0.2, out_dim=1):
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

            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


# =========================================================
# 3b. Auxiliary per-view head
# =========================================================
class AuxHead(nn.Module):
    """Single-view auxiliary classifier for per-backbone supervision.

    Accepts pooled (B, C) or spatial (B, C, H, W). Returns logits (B,).
    """
    def __init__(self, in_dim, hidden_dim=256, dropout_p=0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        if x.dim() == 4:
            x = self.pool(x).flatten(1)
        return self.mlp(x).squeeze(1)


# =========================================================
# 4. Lumbar Single-Task Model
# =========================================================
class ScoliosisLumbarModel(nn.Module):
    def __init__(
        self,
        pa_backbone='resnet18.tv_in1k',
        lat_backbone='resnet18.tv_in1k',
        clinical_dim=5,
        hidden_dim=256,
        dropout_p=0.2,
        fusion_type='projection',   # concat, projection, film, gated, cross_attn, self_attn
        num_heads=8,
        num_transformer_layers=2,
        view_dropout_p=0.0,         # PA view dropout: zero out X_PA with probability p during training
        use_aux_heads=False,        # per-view auxiliary classifier heads
        aux_hidden_dim=256,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.view_dropout_p = view_dropout_p
        self.use_aux_heads = use_aux_heads

        # ViT/DINOv2/TXV backbones have no spatial feature map — block spatial fusions
        pa_source, pa_name, _ = _parse_model_spec(pa_backbone)
        is_vit    = pa_name.lower().startswith(_VIT_PREFIXES)
        is_no_spatial = pa_source in ('dinov2', 'txv') or is_vit
        if is_no_spatial and fusion_type in ('cross_attn', 'self_attn'):
            raise ValueError(
                f"Backbone '{pa_backbone}' has no spatial feature map. "
                f"Cannot use fusion_type='{fusion_type}'. "
                f"Use: concat | projection | film | gated"
            )

        # attention fusion requires spatial feature maps
        self.requires_spatial = fusion_type in ['cross_attn', 'self_attn']
        # plane-aware fusion needs pa/lat features kept separate (not pre-concatenated)
        self.requires_separate_views = fusion_type in ['plane_aware_film']
        pool_flag = not self.requires_spatial

        # image encoders
        self.pa_model  = BackboneModel(pa_backbone,  pool=pool_flag)
        self.lat_model = BackboneModel(lat_backbone, pool=pool_flag)

        pa_dim = self.pa_model.feature_dim
        lat_dim = self.lat_model.feature_dim
        image_fused_dim = pa_dim + lat_dim

        # normalize image features when neither attention nor plane-aware
        if not self.requires_spatial and not self.requires_separate_views:
            self.img_fusion = nn.Sequential(
                nn.LayerNorm(image_fused_dim)
            )

        self.fusion = self._build_fusion_module(
            fusion_type=fusion_type,
            img_dim=image_fused_dim,
            clin_dim=clinical_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            pa_dim=pa_dim,
            lat_dim=lat_dim,
        )

        self.classifier = LumbarClassifier(
            input_dim=self.fusion.out_dim,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
            out_dim=1
        )

        # per-view auxiliary heads (optional)
        if use_aux_heads:
            self.aux_head_pa  = AuxHead(pa_dim,  aux_hidden_dim, dropout_p)
            self.aux_head_lat = AuxHead(lat_dim, aux_hidden_dim, dropout_p)

    def _build_fusion_module(self, fusion_type, img_dim, clin_dim, num_heads=8, num_layers=2,
                              pa_dim=None, lat_dim=None):
        if fusion_type == 'concat':
            return SimpleConcatFusion(img_dim, clin_dim)
        elif fusion_type == 'projection':
            return ProjConcatFusion(img_dim, clin_dim, proj_dim=64)
        elif fusion_type == 'film':
            return FiLMFusion(img_dim, clin_dim)
        elif fusion_type == 'plane_aware_film':
            return PlaneAwareFiLM(pa_dim, lat_dim, clin_dim)
        elif fusion_type == 'gated':
            return GatedFusion(img_dim, clin_dim)
        elif fusion_type == 'cross_attn':
            return CrossAttentionFusion(img_dim, clin_dim, num_heads=num_heads)
        elif fusion_type == 'self_attn':
            return SelfAttentionFusion(
                img_dim,
                clin_dim,
                num_heads=num_heads,
                num_layers=num_layers
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def _apply_view_dropout(self, pa_feat):
        """Zero out X_PA per sample during training (view dropout)."""
        if not self.training or self.view_dropout_p <= 0.0:
            return pa_feat

        B = pa_feat.shape[0]
        keep = torch.bernoulli(
            torch.full((B,), 1.0 - self.view_dropout_p, device=pa_feat.device)
        )
        if pa_feat.dim() == 2:
            mask = keep.unsqueeze(1)
        else:
            mask = keep.view(B, 1, 1, 1)
        return pa_feat * mask

    def forward(self, pa_data, lat_data, clinical, return_aux=False):
        # 1. feature extraction
        pa_feat = self.pa_model(pa_data)
        lat_feat = self.lat_model(lat_data)

        # 2. aux logits — use raw backbone features before view dropout
        if return_aux and self.use_aux_heads:
            aux_pa_logit  = self.aux_head_pa(pa_feat)
            aux_lat_logit = self.aux_head_lat(lat_feat)

        # 3. view dropout — stochastically drop PA during training (main path only)
        pa_feat_main = self._apply_view_dropout(pa_feat)

        # 4. PA/LAT fusion + 5. image-clinical fusion
        if self.requires_separate_views:
            fused_feat = self.fusion(pa_feat_main, lat_feat, clinical)
        else:
            if self.requires_spatial:
                img_fused = torch.cat([pa_feat_main, lat_feat], dim=1)
            else:
                img_fused = torch.cat([pa_feat_main, lat_feat], dim=1)
                img_fused = self.img_fusion(img_fused)
            fused_feat = self.fusion(img_fused, clinical)

        # 6. classification
        out = self.classifier(fused_feat).squeeze(1)        # (B,)

        if return_aux and self.use_aux_heads:
            return out, aux_pa_logit, aux_lat_logit
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 4
    pa = torch.randn(batch_size, 3, 224, 224).to(device)
    lat = torch.randn(batch_size, 3, 224, 224).to(device)
    clinical = torch.randn(batch_size, 5).to(device)

    fusion_types = ['concat', 'projection', 'film', 'gated', 'cross_attn', 'self_attn']

    print("=== Lumbar Model Test (with aux heads) ===")
    for fusion_type in fusion_types:
        model = ScoliosisLumbarModel(
            pa_backbone='resnet18.tv_in1k',
            lat_backbone='resnet18.tv_in1k',
            clinical_dim=5,
            hidden_dim=256,
            dropout_p=0.2,
            fusion_type=fusion_type,
            use_aux_heads=True,
            aux_hidden_dim=128,
        ).to(device)

        out = model(pa, lat, clinical)
        out_aux = model(pa, lat, clinical, return_aux=True)
        print(
            f"[{fusion_type:>11}] main={tuple(out.shape)}  "
            f"aux=({tuple(out_aux[0].shape)}, {tuple(out_aux[1].shape)}, {tuple(out_aux[2].shape)})"
        )
