import torch.nn as nn
from torchvision import models

NUM_CLASSES = 3
ARCHS = ["resnet50", "efficientnet_b0"]


def build_model(name, pretrained=True):
    if name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"modelo desconhecido: {name}")
    return m


def param_groups(model, name, lr_backbone=1e-4, lr_head=1e-3):
    head = model.fc if name == "resnet50" else model.classifier[1]
    head_params = list(head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [{"params": backbone, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head}]