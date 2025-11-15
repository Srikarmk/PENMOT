# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as T
# from torchvision.ops import roi_align
# import numpy as np
#
#
# class DetectionEncoder(nn.Module):
#     """
#     Extract appearance features from detection bounding boxes using ResNet-50.
#     """
#
#     def __init__(self, feature_dim=512, pretrained=True, freeze_backbone=False):
#         """
#         Args:
#             feature_dim: Output feature dimension
#             pretrained: Use ImageNet pretrained weights
#             freeze_backbone: Freeze ResNet weights (for faster training)
#         """
#         super(DetectionEncoder, self).__init__()
#
#         self.feature_dim = feature_dim
#
#         # Load pretrained ResNet-50
#         resnet = models.resnet50(pretrained=pretrained)
#
#         # Remove final FC layer (we'll add our own projection)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Output: [B, 2048, 1, 1]
#
#         # Freeze backbone if requested
#         if freeze_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False
#
#         # Projection head to reduce dimensionality
#         self.projection = nn.Sequential(
#             nn.Linear(2048, feature_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(feature_dim, feature_dim)
#         )
#
#         # Image normalization (ImageNet stats)
#         self.normalize = T.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#
#         print(f"[OK] DetectionEncoder initialized: ResNet-50 -> {feature_dim}D features")
#         if pretrained:
#             print("  Using ImageNet pretrained weights")
#         if freeze_backbone:
#             print("  Backbone frozen")
#
#     def crop_and_resize(self, img, boxes, output_size=(224, 224)):
#         """
#         Crop bounding boxes from image and resize to fixed size.
#
#         Args:
#             img: [3, H, W] image tensor (already normalized)
#             boxes: [N, 4] bounding boxes (x1, y1, x2, y2)
#             output_size: Output size for crops
#
#         Returns:
#             crops: [N, 3, output_size[0], output_size[1]]
#         """
#         device = img.device
#         img = img.unsqueeze(0)  # [1, 3, H, W]
#
#         crops = []
#
#         for box in boxes:
#             x1, y1, x2, y2 = box
#
#             # Add padding to context (10% on each side)
#             w, h = x2 - x1, y2 - y1
#             pad_w, pad_h = w * 0.1, h * 0.1
#
#             x1 = max(0, x1 - pad_w)
#             y1 = max(0, y1 - pad_h)
#             x2 = min(img.shape[3], x2 + pad_w)
#             y2 = min(img.shape[2], y2 + pad_h)
#
#             # Crop and resize
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             crop = img[:, :, y1:y2, x1:x2]
#
#             # Resize to fixed size
#             crop = T.functional.resize(crop, output_size)
#             crops.append(crop[0])
#
#         if len(crops) == 0:
#             return torch.zeros((0, 3, *output_size), device=device)
#
#         crops = torch.stack(crops)  # [N, 3, H, W]
#         return crops
#
#     def forward(self, img, boxes):
#         """
#         Extract features from detection boxes.
#
#         Args:
#             img: [3, H, W] image tensor (already normalized by dataset)
#             boxes: [N, 4] bounding boxes (x1, y1, x2, y2)
#
#         Returns:
#             features: [N, feature_dim] appearance features
#         """
#         if len(boxes) == 0:
#             return torch.zeros((0, self.feature_dim), device=img.device)
#
#         # Crop and resize boxes
#         crops = self.crop_and_resize(img, boxes)  # [N, 3, 224, 224]
#
#         # Extract features with ResNet
#         with torch.set_grad_enabled(self.training):
#             features = self.backbone(crops)  # [N, 2048, 1, 1]
#             features = features.squeeze(-1).squeeze(-1)  # [N, 2048]
#
#             # Project to lower dimension
#             features = self.projection(features)  # [N, feature_dim]
#
#             # L2 normalize
#             features = nn.functional.normalize(features, p=2, dim=1)
#
#         return features
#
#
# class SimpleDetectionEncoder(nn.Module):
#     """
#     Simplified version that extracts features without projection.
#     Use this if you want raw ResNet features.
#     """
#
#     def __init__(self, pretrained=True):
#         super(SimpleDetectionEncoder, self).__init__()
#
#         resnet = models.resnet50(pretrained=pretrained)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])
#
#         self.normalize = T.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#
#     def forward(self, crops):
#         """
#         Args:
#             crops: [N, 3, 224, 224] cropped and resized boxes
#         Returns:
#             features: [N, 2048] raw ResNet features
#         """
#         features = self.backbone(crops)
#         features = features.squeeze(-1).squeeze(-1)
#         return nn.functional.normalize(features, p=2, dim=1)

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.ops import roi_align
import numpy as np


class DetectionEncoder(nn.Module):
    def __init__(self, feature_dim=512, pretrained=True, freeze_backbone=False):
        super(DetectionEncoder, self).__init__()

        self.feature_dim = feature_dim

        resnet = models.resnet50(pretrained=pretrained)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print(f"[OK] DetectionEncoder initialized: ResNet-50 -> {feature_dim}D features")
        if pretrained:
            print("  Using ImageNet pretrained weights")
        if freeze_backbone:
            print("  Backbone frozen")

    def crop_and_resize(self, img, boxes, output_size=(224, 224)):
        device = img.device
        img = img.unsqueeze(0)

        H, W = img.shape[2], img.shape[3]
        crops = []

        for box in boxes:
            x1, y1, x2, y2 = box

            w, h = x2 - x1, y2 - y1

            if w <= 0 or h <= 0:
                crops.append(torch.zeros((3, *output_size), device=device))
                continue

            pad_w, pad_h = w * 0.1, h * 0.1

            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(W, x2 + pad_w)
            y2_pad = min(H, y2 + pad_h)

            x1_pad = int(x1_pad)
            y1_pad = int(y1_pad)
            x2_pad = int(x2_pad)
            y2_pad = int(y2_pad)

            if x2_pad <= x1_pad or y2_pad <= y1_pad:
                crops.append(torch.zeros((3, *output_size), device=device))
                continue

            crop = img[:, :, y1_pad:y2_pad, x1_pad:x2_pad]

            if crop.shape[2] == 0 or crop.shape[3] == 0:
                crops.append(torch.zeros((3, *output_size), device=device))
                continue

            crop = T.functional.resize(crop, output_size)
            crops.append(crop[0])

        if len(crops) == 0:
            return torch.zeros((0, 3, *output_size), device=device)

        crops = torch.stack(crops)
        return crops

    def forward(self, img, boxes):
        if len(boxes) == 0:
            return torch.zeros((0, self.feature_dim), device=img.device)

        crops = self.crop_and_resize(img, boxes)

        with torch.set_grad_enabled(self.training):
            features = self.backbone(crops)
            features = features.squeeze(-1).squeeze(-1)

            features = self.projection(features)

            features = nn.functional.normalize(features, p=2, dim=1)

        return features


class SimpleDetectionEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(SimpleDetectionEncoder, self).__init__()

        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, crops):
        features = self.backbone(crops)
        features = features.squeeze(-1).squeeze(-1)
        return nn.functional.normalize(features, p=2, dim=1)