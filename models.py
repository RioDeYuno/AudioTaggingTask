#models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# Utility functions for initialization
def init_layer(layer, nonlinearity="leaky_relu"):
    """
    Initialize layers with Kaiming Uniform for better training stability.
    """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    """
    Initialize BatchNorm layers.
    """
    bn.bias.data.fill_(0.0)
    bn.running_mean.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_var.data.fill_(1.0)


# Attention Mechanisms
class SpatialAttention2d(nn.Module):
    def __init__(self, in_channels):
        """
        Spatial Attention Block for focusing on significant regions.
        """
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.sigmoid(self.squeeze(x))
        return x * attention_map


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Channel Attention Block for emphasizing important channels.
        """
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.sigmoid(self.fc2(self.relu(self.fc1(self.global_avg_pool(x)))))
        return x * attention_map


# Custom CNN Model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Basic Convolution Block with BatchNorm and ReLU.
        """
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=80):
        """
        Custom CNN with attention mechanisms and adaptive pooling.
        """
        super(CustomCNN, self).__init__()
        self.block1 = ConvBlock(1, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.spatial_attention = SpatialAttention2d(256)
        self.channel_attention = ChannelAttention(256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Inception v3 Model Integration
class InceptionV3(nn.Module):
    def __init__(self, pretrained=True, num_classes=80):
        """
        Inception v3 model adapted for audio classification.
        """
        super(InceptionV3, self).__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=pretrained)
        self.model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Model Selector
def get_model(model_name, num_classes=80):
    """
    Returns the model instance based on the provided name.
    """
    if model_name == "CustomCNN":
        return CustomCNN(num_classes=num_classes)
    elif model_name == "InceptionV3":
        return InceptionV3(pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
