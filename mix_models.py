#Libraries needed to code the model
import torch
from torch import nn
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

#Convolution 3x3 layer
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

#Convolutional 1x1 layer
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#Basic Block for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,alpha_b = 0.9, alpha_g = 0.1,downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        #if norm_layer is None:
        #    norm_layer = nn.BatchNorm2d
        #if groups != 1 or base_width != 64:
        #    raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        #if dilation > 1:
        #    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.alpha_b = alpha_b
        self.alpha_g = alpha_g

        batch_norm = nn.BatchNorm2d
        group_norm = GroupNorm32

        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = norm_layer(planes)

        self.nlb1 = batch_norm(planes)
        self.nlg1 = group_norm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = norm_layer(planes)

        self.nlb2 = batch_norm(planes)
        self.nlg2 = group_norm(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out1 = self.nlb1(out)
        out2 = self.nlg1(out)
        out = out1*self.alpha_b + out2*self.alpha_g
        out = self.relu(out)
        out = self.conv2(out)
        out1 = self.nlb2(out)
        out2 = self.nlg2(out)
        out = out1*self.alpha_b + out2*self.alpha_g


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#Bottleneck Component for ResNet
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, alpha_b = 0.9, alpha_g = 0.1):
        super(Bottleneck, self).__init__()
        #if norm_layer is None:
        #    norm_layer = nn.BatchNorm2d
        
        self.alpha_b = alpha_b
        self.alpha_g = alpha_g

        batch_norm = nn.BatchNorm2d
        group_norm = GroupNorm32
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        #self.bn1 = norm_layer(width)
        self.nlb1 = batch_norm(width)
        self.nlg1 = group_norm(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        #self.bn2 = norm_layer(width)
        self.nlb2 = batch_norm(width)
        self.nlg2 = group_norm(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        #self.bn3 = norm_layer(planes * self.expansion)
        self.nlb3 = batch_norm(planes * self.expansion)
        self.nlg3 = group_norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out1 = self.nlb1(out)
        out2 = self.nlb2(out)
        out1 = torch.mul(out1,self.alpha_b)
        out2 = torch.mul(out2,self.alpha_g)
        out = torch.add(out1,out2)
        out = self.relu(out)

        out = self.conv2(out)
        out1 = self.nlb2(out)
        out2 = self.nlg2(out)
        out1 = torch.mul(out1,self.alpha_b)
        out2 = torch.mul(out2,self.alpha_g)
        out = torch.add(out1,out2)
        out = self.relu(out)

        out = self.conv3(out)
        out1 = self.nlb3(out)
        out2 = self.nlg3(out)
        out1 = torch.mul(out1,self.alpha_b)
        out2 = torch.mul(out2,self.alpha_g)
        out = torch.add(out1,out2)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ResNet50 architecture for CIFAR-10 (images of size 32*32*3)
class ResNet(nn.Module):

    def __init__(self, block, layers,alpha_b = 0.9, alpha_g = 0.1, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(ResNet, self).__init__()
        self.batch_norm = nn.BatchNorm2d
        self.group_norm = GroupNorm32

        self.alpha_b = alpha_b
        self.alpha_g = alpha_g

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.nlb1 = self.batch_norm(self.inplanes)
        self.nlg1 = self.group_norm(self.inplanes)
                
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, GroupNorm32)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.nlb3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.nlb2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        batch_norm = self.batch_norm
        group_norm = self.group_norm
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                batch_norm(planes * block.expansion),group_norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, alpha_b= self.alpha_b, alpha_g= self.alpha_g))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, alpha_b = self.alpha_b, alpha_g= self.alpha_g ))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x1 = self.nlb1(x)
        x2 = self.nlg1(x)
        x = x1 * self.alpha_b + x2 * self.alpha_g
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, alpha_b = 0.9, alpha_g = 0.1 ,**kwargs):
    model = ResNet(block, layers, alpha_b = alpha_b, alpha_g = alpha_g, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet50(pretrained=False, progress=True, alpha_b = 0.9, alpha_g = 0.1, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, alpha_b = alpha_b, alpha_g = alpha_g,
                   **kwargs)

class GroupNorm32(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=2, **kargs):
        super().__init__(num_groups, num_channels, **kargs)

def ResNet50(alpha_b = 0.9, alpha_g = 0.1):
    return resnet50(pretrained = False,alpha_b = alpha_b, alpha_g = alpha_g)