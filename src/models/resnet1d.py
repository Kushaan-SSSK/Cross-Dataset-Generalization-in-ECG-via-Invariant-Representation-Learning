
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet1d(nn.Module):
    def __init__(self, input_channels=12, num_classes=2, layers=[2, 2, 2, 2], planes=[64, 128, 256, 512]):
        super(ResNet1d, self).__init__()
        import logging
        logging.getLogger(__name__).info(f"DEBUG: ResNet1d initialized with num_classes={num_classes}")
        self.inplanes = planes[0]
        
        # Initial convolution
        # Input: (B, 12, 1000)
        self.conv1 = nn.Conv1d(input_channels, planes[0], kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock1d, planes[0], layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock1d, planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, planes[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(planes[3] * BasicBlock1d.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feats=False):
        """
        Args:
            x: Input tensor (B, C, L)
            return_feats: If True, returns (logits, features) tuple.
        """
        # x: (B, C, L)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feats = x.view(x.size(0), -1)
        out = self.fc(feats)

        if return_feats:
            return out, feats
        return out
