import torch.nn as nn

__all__ = ['resnet32_ln']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, in_shape=None):
        working_shape = in_shape
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        working_shape = (planes, working_shape[1] // stride, working_shape[2] // stride)
        self.ln1 = nn.LayerNorm(list(working_shape))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = nn.LayerNorm(list(working_shape))
        self.downsample = downsample
        self.output_shape = working_shape

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ln2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_shape=(3, 32, 32)):
        self.working_shape = in_shape
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.working_shape = (16, self.working_shape[1], self.working_shape[2])
        self.ln1 = nn.LayerNorm(list(self.working_shape))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample(self.inplanes, planes * block.expansion, stride=stride)
        layers = []
        _block = block(self.inplanes, planes, stride, downsample, in_shape=self.working_shape)
        layers.append(_block)
        self.working_shape = _block.output_shape
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            _block = block(self.inplanes, planes, in_shape=self.working_shape)
            layers.append(_block)
            self.working_shape = _block.output_shape
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Downsample(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.planes = planes
        self.stride = stride

    def forward(self, x):
        return self.net(x)


def resnet32_ln(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    # change n=3 for ResNet-20, and n=9 for ResNet-56
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model
