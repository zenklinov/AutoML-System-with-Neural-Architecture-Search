import torch
import torch.nn as nn
import torch.nn.functional as F

class SearchableConvBlock(nn.Module):
    """
    A unified block that can instantiate different operations based on config.
    Search Space:
    - Kernel Size: 3x3, 5x5
    - Out Channels: defined by multiplier
    - Activation: ReLU, SiLU
    - Residual: True/False
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_name, use_residual):
        super(SearchableConvBlock, self).__init__()
        self.use_residual = use_residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Padding to keep spatial dimensions same
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation_name == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation_name == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()
            
        # Projection for residual connection if dimensions change
        self.project = None
        if use_residual and (in_channels != out_channels):
            self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        
        if self.use_residual:
            if self.project:
                identity = self.project(identity)
            out += identity
            
        return out

class SuperNet(nn.Module):
    """
    The SuperNet generated from a specific configuration.
    In a Weight Sharing NAS (One-Shot), this would be much larger.
    For this 'Multi-Trial' demo, we instantiate discrete models.
    """
    def __init__(self, config, num_classes=10):
        super(SuperNet, self).__init__()
        
        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        layers = []
        in_c = 32
        
        # Dynamic Layer Generation based on config
        # config example: {"num_layers": 3, "l1_kernel": 3, "l1_out": 64...}
        num_layers = config.get("num_layers", 3)
        
        for i in range(num_layers):
            k_key = f"l{i}_kernel"
            c_key = f"l{i}_out"
            act_key = f"l{i}_act"
            res_key = f"l{i}_res"
            
            kernel = config.get(k_key, 3)
            out_c = config.get(c_key, 64)
            act = config.get(act_key, "relu")
            res = config.get(res_key, True)
            
            layers.append(SearchableConvBlock(in_c, out_c, kernel, act, res))
            layers.append(nn.MaxPool2d(2)) # Simple downsampling strategy
            in_c = out_c
            
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
