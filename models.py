from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

class TinyCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

class FasterRCNNBackboneClassifier(nn.Module):
    """
    Use only detection backbone (ResNet+FPN) -> fixed feature vector via pooling -> linear head.
    Output remains classification logits.
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # build a detection model to get its backbone (ResNet50-FPN)
        det = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None
        )
        self.backbone = det.backbone  # returns dict of feature maps at multiple pyramid levels
        # Use a simple strategy: pick highest-res FPN level ("0") when present; else first key.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # ResNet50-FPN has 256 channels for each FPN output
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        feats = self.backbone(x)  # OrderedDict[str, Tensor]
        if isinstance(feats, dict):
            key = "0" if "0" in feats else list(feats.keys())[0]
            f = feats[key]
        else:
            f = feats
        v = self.pool(f).flatten(1)  # [B,256]
        return self.head(v)

def make_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    arch = arch.lower()

    if arch in ["tinycnn", "custom", "custom_cnn"]:
        return TinyCNN(num_classes)

    if arch in ["convnext_tiny", "convnext"]:
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = convnext_tiny(weights=weights)
        # replace classification head
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
        return m

    if arch in ["vit_b_16", "vit", "vit_base"]:
        try:
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            m = vit_b_16(weights=weights)
            in_f = m.heads.head.in_features
            m.heads.head = nn.Linear(in_f, num_classes)
            return m
        except Exception:
            # fallback to timm if torchvision ViT unavailable
            import timm
            m = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
            return m

    if arch in ["fasterrcnn_backbone", "faster_rcnn_backbone", "frcnn_backbone"]:
        return FasterRCNNBackboneClassifier(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(
        f"Unsupported arch: {arch}. Use tinycnn|convnext_tiny|vit_b_16|fasterrcnn_backbone"
    )
