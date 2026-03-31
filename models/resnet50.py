import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import Bottleneck

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Clamp p to avoid unstable behavior when p gets too small during training.
        p = self.p.clamp(min=1e-3)
        return self.gem(x, p, self.eps)

    @staticmethod
    def gem(x, p, eps):
        return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), (1, 1)).pow(1. / p)

class FineGrainedModel(nn.Module):
    def __init__(self, num_classes=100, dropout=0.5, gem_p=3.0):
        super(FineGrainedModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.gem = GeM(p=gem_p)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.gem(x)
        logits = self.head(x)
        return logits
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 1. Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. Excitation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # 輸出 0~1 之間的權重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SEBottleneck(Bottleneck):
    """ 繼承原始的 Bottleneck 並加入 SE 層 """
    def __init__(self, *args, **kwargs):
        super(SEBottleneck, self).__init__(*args, **kwargs)
        self.se = SELayer(self.bn3.num_features)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
def get_resnet50():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(2048, 100)
    return model

def get_se_resnet50(num_classes=100):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    state_dict = model.state_dict()
    model = models.ResNet(SEBottleneck, [3, 4, 6, 3])
    model.load_state_dict(state_dict, strict=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def get_finegrained_resnet50(num_classes=100, dropout=0.5, gem_p=3.0):
    return FineGrainedModel(num_classes=num_classes, dropout=dropout, gem_p=gem_p)

if __name__ == "__main__":
    model = get_resnet50()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Parameters: {params:.2f}M")